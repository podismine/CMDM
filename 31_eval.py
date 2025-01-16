import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import Task1Data
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import copy, shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    def __init__(self, input_dim=64, hidden_dim=128, time_dim=256, condition_dim=64):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial projection
        self.init_proj = nn.Linear(input_dim, hidden_dim)
        
        # Condition processing
        self.condition_proj = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # *2 because we concatenate data and mask
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            # nn.Linear(hidden_dim + condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.condition_embed = ConditionalEmbedding(condition_dim)
        # Encoder
        self.down1 = DownBlock1D(hidden_dim, hidden_dim * 2)
        self.down2 = DownBlock1D(hidden_dim * 2, hidden_dim * 4)
        
        # Bottleneck
        self.bottleneck = ResnetBlock1D(hidden_dim * 4, hidden_dim * 4)
        
        # Decoder
        self.up1 = UpBlock1D(hidden_dim * 4, hidden_dim * 2)
        self.up2 = UpBlock1D(hidden_dim * 2, hidden_dim)
        
        # Output
        self.final_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, condition, t, age, sex):
        # Time embedding
        t = self.time_mlp(t)
        
        # Initial projection
        x = self.init_proj(x)  # (B, hidden_dim)
        cond_emb = self.condition_embed(age, sex)
        # Process condition
        # cond = self.condition_proj(condition)  # (B, hidden_dim)

        cond = self.condition_fusion(cond_emb
            # torch.cat([cond, cond_emb], dim=1)
        )
        x = x + cond  # Add condition information
        
        # Encoder
        skip1 = x
        x = self.down1(x, t)
        skip2 = x
        x = self.down2(x, t)
        
        # Bottleneck
        x = self.bottleneck(x, t)
        
        # Decoder with skip connections
        x = self.up1(x, skip2, t)
        x = self.up2(x, skip1, t)
        
        # Output
        return self.final_layer(x)

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
from tqdm import tqdm
class MaskedDDPM1D(nn.Module):
    def __init__(self, input_dim=64, beta_start=1e-4, beta_end=0.02, n_timesteps=1000):
        super().__init__()
        # 定义noise schedule
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).cuda()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).cuda()
        self.device='cuda'
        # 定义模型
        self.model = ConditionalUNet1D(input_dim=input_dim).cuda()
        
        # def forward(self, x, mask, t):
        #     # mask shape: (B, 64)
        #     # x shape: (B, 64)
        #     condition = torch.cat([x * mask, mask], dim=1)  # (B, 128)
        #     return self.model(x, condition, t)
        # population loss
        self.num_classes = 100
        self.lambda_moment = 1.0
        self.lambda_dist = 0.1
        self.lambda_cov = 0.1
        self.min_samples = 2
        
        # 维护每个类别的滑动统计量
        self.register_buffer('running_mean', torch.zeros(2,100, 64))
        self.register_buffer('running_var', torch.ones(2,100, 64))
        self.register_buffer('running_skew', torch.zeros(2,100, 64))
        self.register_buffer('running_kurt', torch.zeros(2,100, 64))
        self.register_buffer('running_cov', torch.zeros(2,100, 64, 64))
        self.register_buffer('update_counts', torch.zeros(2,100))
        self.momentum = 0.1
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

def train_step(model, optimizer, x, mask, age, sex, device):
    batch_size = x.shape[0]
    
    t = torch.randint(0, 1000, (batch_size,), device=device)
    noise = torch.randn_like(x).to(device)
    
    alphas_cumprod = model.alphas_cumprod[t][:, None]
    noisy_sample = torch.sqrt(alphas_cumprod) * x + torch.sqrt(1 - alphas_cumprod) * noise
    noise_pred = model(noisy_sample, mask, t, age, sex)
    
    loss = F.mse_loss(noise_pred, noise)
    
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), grad_norm.item()
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="fmri tokenizer")

    parser.add_argument('-m', "--mask", type=float)
    parser.add_argument('-s', "--seed", type=int, default=42)
    parser.add_argument('-l', "--lam", type=float, default=0.01)
    parser.add_argument("--name", type=str,default='run_final')
    parser.add_argument("--mo", type=float, default=0.1)

    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    name = f"search_param{args.lam}_mom{args.mo}"
    log_dir = f"logs_search/log_{name}"
    checkpoint_dir = f"checkpoint/checkpoint_{name}"
    model = MaskedDDPM1D(input_dim=64).cuda()

    checkpoint = torch.load(f"{checkpoint_dir}/last_model.pth", map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint)
    model.cuda().eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for sex in [0,1]:
        for age in range(100):
            print(f"Conducting... {sex} {age}")
            batch = 512
            age_torch = torch.Tensor([age]*batch).to(device,non_blocking=True).float().view(-1,1) / 100
            sex_torch = torch.Tensor([sex]*batch).to(device,non_blocking=True).long().view(-1)

            mask = torch.ones((batch, 64)).to(device,non_blocking=True)
            samples = model.sample(mask, age_torch, sex_torch, device = device)

            to_save = samples.detach().cpu().numpy()
            save_root = f'res/{name}'
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            np.save(f"{save_root}/large_sex-{sex}_age-{age}.npy",to_save)

            del mask, samples