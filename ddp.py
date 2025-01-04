import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from tqdm import tqdm

# class PositionalEmbedding(nn.Module):
#     def __init__(self, max_len: int = 64):
#         super().__init__()
#         self.position_embedding = torch.nn.Embedding(max_len, embed_dim)

#     def forward(self, x):
#         seq_len = x.size(1)
#         position_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)  # (1, seq_len)
#         pos_embed = self.position_embedding(position_ids)  # (1, seq_len, embed_dim)
#         return x + pos_embed
    
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class DownBlock1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.resnet = ResnetBlock1D(in_dim, out_dim)
        self.downsample = nn.Linear(out_dim, out_dim)
        
    def forward(self, x, t):
        x = self.resnet(x, t)
        x = self.downsample(x)
        return x

class UpBlock1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.resnet = ResnetBlock1D(in_dim + out_dim, out_dim)
        self.upsample = nn.Linear(in_dim, in_dim)
        
    def forward(self, x, skip, t):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=-1)
        x = self.resnet(x, t)
        return x

class ResnetBlock1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.time_mlp = nn.Linear(256, out_dim)
        
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t)
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)

class ConditionalEmbedding(nn.Module):
    def __init__(self, conditional_dim=64):
        super().__init__()
        # Age embedding
        self.age_embed = nn.Sequential(
            nn.Linear(1, conditional_dim // 2),
            nn.GELU(),
            nn.Linear(conditional_dim // 2, conditional_dim)
        )
        
        # Sex embedding (2类别)
        self.sex_embed = nn.Embedding(2, conditional_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(conditional_dim * 2, conditional_dim),
            nn.GELU(),
            nn.Linear(conditional_dim, conditional_dim)
        )
        
    def forward(self, age, sex):
        # age: (B, 1), normalized age value
        # sex: (B,), binary value (0 or 1)
        age_emb = self.age_embed(age)
        sex_emb = self.sex_embed(sex)
        
        # 融合条件信息
        conditional = torch.cat([age_emb, sex_emb], dim=1)
        return self.fusion(conditional)

class ConditionalUNet1D(nn.Module):
    def __init__(
        self, 
        input_dim: int = 64,
        hidden_dims: List[int] = [128, 256, 512],  # 每层的隐藏维度
        time_dim: int = 256,
        condition_dim: int = 64,
        num_res_blocks: int = 2,  # 每层的残差块数量
    ):
        super().__init__()
        self.time_dim = time_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.depth = len(hidden_dims)
        # self.pos_embed = PositionalEmbedding(max_len=input_dim)
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Conditional embedding
        self.condition_embed = ConditionalEmbedding(condition_dim)
        
        # Initial projection
        # self.init_proj = nn.Linear(input_dim, hidden_dims[0])
        self.init_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        # Mask and data condition processing
        self.mask_condition_proj = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # Condition fusion layer
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dims[0] + condition_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # 动态创建编码器层
        self.down_layers = nn.ModuleList()
        for i in range(self.depth - 1):
            for _ in range(num_res_blocks):
                self.down_layers.append(
                    ResnetBlock1D(hidden_dims[i], hidden_dims[i])
                )
            self.down_layers.append(
                DownBlock1D(hidden_dims[i], hidden_dims[i + 1])
            )
            
        # Bottleneck
        self.bottleneck_layers = nn.ModuleList([
            ResnetBlock1D(hidden_dims[-1], hidden_dims[-1])
            for _ in range(num_res_blocks)
        ])
        
        # 动态创建解码器层
        self.up_layers = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            self.up_layers.append(
                UpBlock1D(hidden_dims[i], hidden_dims[i - 1])
            )
            for _ in range(num_res_blocks):
                self.up_layers.append(
                    ResnetBlock1D(hidden_dims[i - 1], hidden_dims[i - 1])
                )
        
        # Output
        self.final_layer = nn.Linear(hidden_dims[0], input_dim)
        
    def forward(self, x, mask_condition, t, age, sex):

        
        # pos_encoding = self.pos_embed(torch.zeros(x.size(0), 64).to(x.device))  # (B, 64, hidden_dims[0])
        # x = x + pos_encoding

        ################
        # Time embedding
        t = self.time_mlp(t)
        
        # Get conditional embedding
        cond_emb = self.condition_embed(age, sex)
        
        # Initial projection
        x = self.init_proj(x)
        
        # Process mask condition
        mask_cond = self.mask_condition_proj(mask_condition)
        
        # Fuse all conditions
        conditions = self.condition_fusion(
            torch.cat([mask_cond, cond_emb], dim=1)
        )
        x = x + conditions
        
        # 存储skip connections
        skip_connections = []
        
        # Encoder
        down_block_counter = 0
        for i, layer in enumerate(self.down_layers):
            if isinstance(layer, DownBlock1D):
                skip_connections.append(x)
                x = layer(x, t)
                down_block_counter += 1
            else:
                x = layer(x, t)
                
        # Bottleneck
        for layer in self.bottleneck_layers:
            x = layer(x, t)
            
        # Decoder
        up_block_counter = 0
        for i, layer in enumerate(self.up_layers):
            if isinstance(layer, UpBlock1D):
                skip = skip_connections.pop()
                x = layer(x, skip, t)
                up_block_counter += 1
            else:
                x = layer(x, t)
        
        assert len(skip_connections) == 0, "All skip connections should be used"
        assert down_block_counter == up_block_counter, "Number of up and down blocks should match"
        
        return self.final_layer(x)

class MaskedDDPM1D(nn.Module):
    def __init__(self, \
                input_dim=64, \
                beta_start=1e-4, \
                beta_end=0.02, \
                n_timesteps=1000, \
                hidden_dims: List[int] = [128, 256, 512], \
                time_dim: int = 256, \
                condition_dim: int = 64, \
                num_res_blocks: int = 2,
                device = 'cuda'
                ):
        super().__init__()
        # 定义noise schedule
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        
        # 定义模型
        # self.model = ConditionalUNet1D(input_dim=input_dim).cuda()
        self.model = ConditionalUNet1D(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            time_dim=time_dim,
            condition_dim=condition_dim,
            num_res_blocks=num_res_blocks
        ).to(device)
    # def forward(self, x, mask, t):
    #     # mask shape: (B, 64)
    #     # x shape: (B, 64)
    #     condition = torch.cat([x * mask, mask], dim=1)  # (B, 128)
    #     return self.model(x, condition, t)
    def forward(self, x, mask, t, age, sex):
        # x: (B, 64)
        # mask: (B, 64)
        # age: (B, 1)
        # sex: (B,)
        mask_condition = torch.cat([x * mask, mask], dim=1)
        return self.model(x, mask_condition, t, age, sex)
    
    def sample(self, mask,age,sex, n_steps=1000, device="cpu"):
        batch_size = age.shape[0]
        dim = 64
        x = torch.randn(batch_size, dim).to(self.device)
        # 逐步去噪
        with torch.no_grad():
            for t in tqdm(reversed(range(n_steps)), position = 0):
            # for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t_tensor = torch.tensor([t], device=self.device).repeat(batch_size)
                
                # 预测噪声
                noise_pred = self.forward(x, mask, t_tensor,age,sex)
                
                # 计算去噪后的样本
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                    
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
                ) + torch.sqrt(beta) * noise
            
        return x