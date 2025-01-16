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
            nn.Linear(hidden_dim + condition_dim, hidden_dim),
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
        cond = self.condition_proj(condition)  # (B, hidden_dim)

        cond = self.condition_fusion(
            torch.cat([cond, cond_emb], dim=1)
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
        
        # 定义模型
        self.model = ConditionalUNet1D(input_dim=input_dim).cuda()
        
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


class SPL(object):
    def __init__(self, epochs = 500, init_rate = 0.25, end_rate = 1.0):
        super().__init__()
        self.losses = []
        self.init_rate = init_rate
        self.end_rate = end_rate
        self.all_epochs = epochs
    def update(self,losses, cur_epoch):
        self.losses.extend(losses.detach().cpu().tolist())
        loss_tensor = torch.tensor(self.losses)
        cur_rate = min(cur_epoch / self.all_epochs * (self.end_rate - self.init_rate) + self.init_rate, 1.0)
        threshold = torch.quantile(loss_tensor, cur_rate)

        # weights = threshold - losses.detach()
        # weights[weights>0]=1
        # weights[weights<0]=0

        # weights =torch.sigmoid(threshold - losses).detach()
        weights = (threshold - losses.detach()) #/ (torch.quantile(loss_tensor, 1.0) - torch.quantile(loss_tensor, 0.0))
        weights[weights < 0] = 0
        weights = weights / weights.max()
        # -0.5 0 0.5   
        # min = threshold - torch.quantile(loss_tensor, 0.0) 0 - max
        # max = threshold - torch.quantile(loss_tensor, 1.0) -max - 0 
        #weights[weights>0.5] = 1
        # weights = (threshold - losses) / (threshold - torch.quantile(loss_tensor, 0.0))
        # weights = threshold - losses
        # weights[weights < 0] = 0
        # weights[weights > 0] = 1
        # print(weights); exit()
        self.losses = []
        return weights

def train_step(model, optimizer, x, mask, age, sex,spl,epoch, device):
    batch_size = x.shape[0]
    
    t = torch.randint(0, 1000, (batch_size,), device=device)
    noise = torch.randn_like(x).to(device)
    
    alphas_cumprod = model.alphas_cumprod[t][:, None]
    noisy_sample = torch.sqrt(alphas_cumprod) * x + torch.sqrt(1 - alphas_cumprod) * noise
    noise_pred = model(noisy_sample, mask, t, age, sex)
    
    loss_all = F.mse_loss(noise_pred, noise, reduction='none').mean(1)
    weights = spl.update(loss_all, epoch).to(t.device)
    loss = (loss_all * weights).sum() / weights.sum()
    # loss = (loss_all).mean()
    
    optimizer.zero_grad()
    loss.backward()
    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    return loss.item(), grad_norm.item()
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="fmri tokenizer")

    parser.add_argument('-m', "--mask", type=float)
    parser.add_argument('-s', "--seed", type=int, default=42)
    parser.add_argument("--name", type=str,default='run_final')

    args = parser.parse_args()

    return args
if __name__ == "__main__":
    args = parse_args()
    name = f"mask_addSP_run9_spl_norm_rerun2_mask{args.mask}"
    log_dir = f"logs/log_{name}"
    checkpoint_dir = f"checkpoint/checkpoint_{name}"
    writer = SummaryWriter(log_dir=log_dir)

    def try_do(func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")

    try_do(os.makedirs, log_dir)
    try_do(os.makedirs, checkpoint_dir)
    spl = SPL(epochs = 1000, init_rate = 0.25, end_rate = 1.0)
    model = MaskedDDPM1D(input_dim=64).cuda()


    optimizer = torch.optim.AdamW(model.parameters())
    # loss = train_step(model, optimizer, x, mask, device='cpu')
    # samples = model.sample(x, mask)
    from torch.optim.lr_scheduler import MultiStepLR

    scheduler = MultiStepLR(
        optimizer,
        milestones = [1000,1500],  # 在1/3和2/3处降低学习率
        # milestones = [500,1000],  # 在1/3和2/3处降低学习率
        gamma=0.1 #0.001,1e-4,1e-5,1e-6
    )
    train_dataset = Task1Data(is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=512,num_workers=8,shuffle=False)

    best_train_loss = 99999.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask_ratio = args.mask
    cc = 0
    for epoch_counter in range(2000):
        n_steps = 50
        total_train_loss = 0.
        total_val_loss = 0.
        count = 0.

        model.train()
        for step, (x, age, sex) in enumerate(train_loader):
            # padded_tensor = F.pad(tensor, (0, 2))
            x = x.to(device,non_blocking=True).float()[...,0]#[:,None]
            x = F.pad(x,(0,2))

            mask = torch.ones_like(x).to(device,non_blocking=True)
            num_masked_tokens = int(62 * mask_ratio)
            for i in range(x.size(0)):
                masked_indices = torch.randperm(62)[:num_masked_tokens]
                mask[i, masked_indices] = 0

            age = age.to(device,non_blocking=True).float().view(-1,1) / 100
            sex = sex.to(device,non_blocking=True).long().view(-1)

            loss, grad_norm = train_step(model, optimizer, x, mask, age, sex,spl,epoch_counter, device=device)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
            writer.add_scalar('grad_norm', grad_norm, global_step = cc);cc+=1
            total_train_loss += len(x) * float(loss)
            count += len(x)

        scheduler.step()
        total_train_loss /= count
        print(f"[{epoch_counter}]: train_loss: {total_train_loss:.4f}")
        writer.add_scalar('total_train_loss', total_train_loss, global_step=epoch_counter)

        if epoch_counter % 100 == 0:
            torch.save(model.state_dict(), os.path.join(f"{checkpoint_dir}/{epoch_counter}_model.pth"))

        if best_train_loss > total_train_loss:
            best_train_loss = total_train_loss
            torch.save(model.state_dict(), os.path.join(f"{checkpoint_dir}/best_model.pth"))

    torch.save(model.state_dict(), os.path.join(f"{checkpoint_dir}/last_model.pth"))
