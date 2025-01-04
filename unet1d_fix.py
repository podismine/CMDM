# conditional 1d unet model
import math
from pathlib import Path
from functools import partial
from collections import namedtuple
import numpy as np
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, reduce, repeat

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# data

class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)
    
    
class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        context_dim = 512

        self.to_q = nn.Conv1d(dim, hidden_dim, 1, bias = False)
        self.to_kv = nn.Conv1d(context_dim, hidden_dim * 2, 1, bias = False)
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x, context=None):
        b, c, n = x.shape
        if context is None:
            q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        else:
            q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), (q, k, v))

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)
    
class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.LayerNorm(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)
    
# model
class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.x_embed = nn.Linear(1, hidden_size)

        encoder_transformer = nn.TransformerEncoderLayer(d_model = hidden_size, \
                                                         nhead=4, dim_feedforward=hidden_size * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_transformer, num_layers=4, norm = nn.LayerNorm(hidden_size))

        self.proj = MLPHead(hidden_size, hidden_size // 2, 1)
    def forward(self,x):
        x = self.x_embed(x[:,0][...,None])
        x = self.encoder(x)
        x = self.proj(x)[...,0]
        return x[:,None]
    
class Encoder2(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        self.dim = dim
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(1, init_dim, 7, padding = 3)
        self.age_embedding = nn.Linear(1, dim)
        self.sex_embedding = nn.Linear(1, dim)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, CrossAttention(dim_in, dim_head = attn_dim_head, heads = attn_heads))),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, CrossAttention(dim_in, dim_head = attn_dim_head, heads = attn_heads))),
                # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, CrossAttention(dim_out, dim_head = attn_dim_head, heads = attn_heads))),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, CrossAttention(dim_out, dim_head = attn_dim_head, heads = attn_heads))),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x,text_embed = None):

        x = self.init_conv(x)
        r = x.clone()

        t = None

        h = []

        for block1, attn1, block2, attn2, downsample in self.downs:
            x = block1(x, t)
            x = attn1(x, text_embed)
            h.append(x)

            x = block2(x, t)
            x = attn2(x, text_embed)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, text_embed)
        x = self.mid_block2(x, t)

        for block1, attn1, block2, attn2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = attn1(x, text_embed)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn2(x, text_embed)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
class Unet1d(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        fine_tuning= False,
        mask_ratio = 0.2
    ):
        super().__init__()

        # determine dimensions
        self.dim = dim
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels

        self.fine_tuning = fine_tuning

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(4, init_dim, 7, padding = 3)
        self.age_embedding = nn.Linear(1, dim)
        self.sex_embedding = nn.Linear(1, dim)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # mask part
        self.mask_ratio = mask_ratio
        self.mask_embed = nn.Parameter(torch.zeros([1, 1, 1]))
        # self.mask_encoder = Encoder2(64,dim_mults = (1,4)) #Encoder(dim, 256)
        self.mask_encoder = Encoder(hidden_size=256) #Encoder(dim, 256)
        self.mask_decoder = Encoder(hidden_size=256) #Encoder(dim, 256)

        self.encoder_pos = nn.Parameter(torch.zeros(1, 1, dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, 1, dim))
        self.ddpm_pos = nn.Parameter(torch.zeros(1, 1, dim))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, CrossAttention(dim_in, dim_head = attn_dim_head, heads = attn_heads))),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, CrossAttention(dim_in, dim_head = attn_dim_head, heads = attn_heads))),
                # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, CrossAttention(dim_out, dim_head = attn_dim_head, heads = attn_heads))),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, CrossAttention(dim_out, dim_head = attn_dim_head, heads = attn_heads))),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

        self.init_weight()
    def init_weight(self):
        torch.nn.init.normal_(self.encoder_pos, std=.02)
        torch.nn.init.normal_(self.ddpm_pos, std=.02)
        torch.nn.init.normal_(self.decoder_pos, std=.02)

    def sample_orders(self,bsz, length):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(length)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).long().cuda()
        return orders
    
    def sample_mask(self, x, B, C):
        mask_list = [f*0.1 for f in range(int(self.mask_ratio // 0.1)+1)] 
        cur_mask_ratio = np.random.choice(mask_list,1)
        num_masked_tokens = int(np.ceil(C * cur_mask_ratio))
        mask = torch.ones(B, 1, C).cuda()
        orders = self.sample_orders(B, C)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                            src=torch.zeros(B, C).cuda())
        mask= mask[:,None]
        
        mask_tokens = self.mask_embed.expand(B, -1, C)

        cond_x = x.clone()
        cond_x = cond_x * mask + (1-mask) * mask_tokens

        return cond_x, mask
    
    def forward(self, x, time, age, sex, text_embed = None, x_self_cond = None):
        B,T,C = x.shape
        mask_list = [f*0.1 for f in range(int(self.mask_ratio // 0.1)+1)] 
        cur_mask_ratio = np.random.choice(mask_list,1)
        mask_tokens = self.mask_embed.expand(B, -1, C)
        # print(f"check: {cur_mask_ratio}")

        if cur_mask_ratio == 0 or not self.training or self.fine_tuning is True:
            # print(f"\r Sampling....", end='', flush=True)
            cond_x = x.clone()
            mask = None
            cond_x += self.encoder_pos
            cond_x = self.mask_encoder(cond_x)
            cond_x += self.decoder_pos
            decoder_x = self.mask_decoder(cond_x)
        else:
            num_masked_tokens = int(np.ceil(self.dim * cur_mask_ratio))
            mask = torch.ones(B, C).cuda()
            orders = self.sample_orders(B, C)
            mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                                src=torch.zeros(B, C).cuda())
            mask= mask[:,None]
            cond_x = x.clone()
            cond_x += self.encoder_pos

            cond_x = cond_x[mask.nonzero(as_tuple=True)].reshape(B,1,-1)
            cond_x = self.mask_encoder(cond_x)
            x_after_pad = mask_tokens.clone()
            x_after_pad[mask.nonzero(as_tuple=True)] = cond_x.reshape(-1)
            x_after_pad += self.decoder_pos
            decoder_x = self.mask_decoder(x_after_pad)
        # cond_x = self.mask_encoder(cond_x)
        age_embedding = self.age_embedding(age)
        sex_embedding = self.sex_embedding(sex)
        # for f in (x, decoder_x, age_embedding, sex_embedding):
        #     print(f.shape)
        # exit()
        decoder_x += self.ddpm_pos
        x = torch.cat((x, decoder_x, age_embedding, sex_embedding), dim = 1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, attn1, block2, attn2, downsample in self.downs:
            x = block1(x, t)
            x = attn1(x, text_embed)
            h.append(x)

            x = block2(x, t)
            x = attn2(x, text_embed)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, text_embed)
        x = self.mid_block2(x, t)

        for block1, attn1, block2, attn2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = attn1(x, text_embed)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn2(x, text_embed)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x =  self.final_conv(x)
        return x , mask
    
    
if __name__ == '__main__':
    model = Unet1d(64)
    x = torch.randn(1, 1, 64)
    y = model(x, torch.Tensor([0]), torch.Tensor([0]).reshape(1,1,1), torch.Tensor([0]).reshape(1,1,1))
    print(y.shape)