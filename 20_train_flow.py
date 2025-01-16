import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return x + 0.1 * self.net(x)  # 缩小残差影响
    
class ConditionalCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, condition_dim, mask_type='checkerboard'):
        super().__init__()
        self.dim = dim
        
        # 使用交替mask模式
        # if mask_type == 'checkerboard':
        self.mask = torch.arange(dim) % 2
        self.mask = self.mask.float().view(1, -1)
        
        # 条件编码器
        self.condition_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 主网络
        masked_input_dim = dim // 2 + hidden_dim  # 修正输入维度
        self.nn = nn.Sequential(
            nn.Linear(masked_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, dim)  # 输出维度为dim
        )
        
        # 初始化为接近恒等变换
        nn.init.zeros_(self.nn[-1].weight)
        nn.init.zeros_(self.nn[-1].bias)
        
    def forward(self, x, condition, reverse=False):
        b, d = x.shape
        mask = self.mask.to(x.device)
        
        # 增强条件编码
        condition = self.condition_net(condition)
        
        # 分离masked和unmasked部分
        x_masked = x * mask
        x_unmasked = x * (1 - mask)
        
        # 获取masked值
        masked_values = x_masked[mask.bool().expand(b, -1)].view(b, -1)
        # 连接条件
        nn_input = torch.cat([masked_values, condition], dim=1)
        
        # 获取网络输出
        s_t = self.nn(nn_input)
        
        # 分割scale和translation，确保维度正确
        scale = s_t[:, :self.dim]
        translation = s_t[:, :self.dim]
        
        # 应用mask并确保scale为正
        scale = F.softplus(scale) * (1 - mask)
        translation = translation * (1 - mask)
        
        if not reverse:
            x_transformed = x_unmasked * scale + translation
            log_det = torch.sum(torch.log(scale + 1e-8), dim=1)
        else:
            x_transformed = (x_unmasked - translation) / (scale + 1e-8)
            log_det = None
            
        return x_masked + x_transformed, log_det if not reverse else x_transformed
class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, dim=64, hidden_dim=256, n_layers=6):  # 增加模型容量
        super().__init__()
        self.dim = dim
        
        # 更强大的条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 增加层数并使用不同的mask模式
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask_type = 'checkerboard' if i % 2 == 0 else 'channel'
            self.layers.append(
                ConditionalCouplingLayer(dim, hidden_dim, hidden_dim, mask_type)
            )
        
    def encode_condition(self, age, gender):
        # Combine and encode conditions
        condition = torch.cat([age.unsqueeze(1), 
                             gender.float().unsqueeze(1)], dim=1)
        return self.condition_encoder(condition)
    
    def forward(self, x, age, gender, reverse=False):
        condition = self.encode_condition(age, gender)
        
        log_det_sum = 0
        if not reverse:
            for layer in self.layers:
                x, log_det = layer(x, condition)
                log_det_sum += log_det
            return x, log_det_sum
        else:
            for layer in reversed(self.layers):
                x, _ = layer(x, condition, reverse=True)
            return x
    
    def sample(self, n_samples, age, gender, device='cuda'):
        z = torch.randn(n_samples, self.dim).to(device)
        x = self.forward(z, age, gender, reverse=True)
        return x
    def sample2(self, n_samples, age, gender, device='cuda', temperature=0.7):
        """添加temperature控制采样"""
        z = torch.randn(n_samples, self.dim).to(device) * temperature
        x = self.forward(z, age, gender, reverse=True)
        return x
    
def flow_loss(model, x, age, gender, beta=0.01):
    z, log_det = model(x, age, gender)
    
    # 基础NLL损失
    nll_loss = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * torch.log(
        torch.tensor(2 * torch.pi).to(z.device)) + log_det
    
    # 添加KL散度正则化
    kl_loss = 0.5 * torch.sum(z**2 + torch.log(1e-8 + torch.std(z, dim=0)**2) - 1, dim=1)
    
    return -(nll_loss.mean() - beta * kl_loss.mean())

def train_step(model, optimizer, x, age, sex, epoch, device):
    # 注意这里修复了之前的bug，现在正确传入sex而不是age
    loss = flow_loss(model, x, age, sex)
    
    optimizer.zero_grad()
    loss.backward()
    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    return loss.item()# , grad_norm.item()


import numpy as np
from scipy import linalg
def calculate_fidV2(real_data, fake_data, mean_=None, var_=None):
    """
    计算 Fréchet Inception Distance
    
    参数:
    real_data: 真实数据
    fake_data: 生成/拟合的数据
    mean_: 预计算的fake均值（可选）
    var_: 预计算的fake协方差（可选）
    
    返回:
    float: FID 分数
    """
    try:
        # 计算真实数据的统计量
        mu_real = np.mean(real_data, axis=0)
        sigma_real = np.cov(real_data, rowvar=False)
        
        # 使用预计算值或计算fake数据的统计量
        if mean_ is None:
            mu_fake = np.mean(fake_data, axis=0)
        else:
            mu_fake = mean_
            
        if var_ is None:
            sigma_fake = np.cov(fake_data, rowvar=False)
        else:
            sigma_fake = var_
            
        # 确保矩阵是对称的
        sigma_real = (sigma_real + sigma_real.T) / 2
        sigma_fake = (sigma_fake + sigma_fake.T) / 2
        
        # 添加小的正则化项以增加数值稳定性
        eps = 1e-6
        sigma_real += np.eye(sigma_real.shape[0]) * eps
        sigma_fake += np.eye(sigma_fake.shape[0]) * eps
        
        # 计算均值差
        diff = mu_real - mu_fake
        mean_diff = np.sum(diff * diff)
        
        # 计算协方差项
        sigma_prod = sigma_real.dot(sigma_fake)
        
        try:
            # 尝试直接计算矩阵平方根
            covmean = linalg.sqrtm(sigma_prod)
            
            # 处理复数结果
            if np.iscomplexobj(covmean):
                if np.abs(covmean.imag).max() > 1e-3:
                    print("警告：计算结果包含显著的虚部")
                covmean = covmean.real
                
        except np.linalg.LinAlgError:
            # 如果直接计算失败，使用特征值分解方法
            eigenvals, eigenvects = np.linalg.eigh(sigma_prod)
            eigenvals = np.maximum(eigenvals, 0)  # 确保非负
            sqrt_eigenvals = np.sqrt(eigenvals)
            covmean = eigenvects.dot(np.diag(sqrt_eigenvals)).dot(eigenvects.T)
        
        # 计算最终的FID分数
        fid = mean_diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        # 确保结果有效
        if np.isnan(fid) or np.isinf(fid):
            raise ValueError("FID计算结果无效")
            
        return float(fid)
        
    except Exception as e:
        print(f"FID计算出错: {str(e)}")
        return float('inf')
    
def calculate_fid(real_data, fake_data,mean_= None, var_ = None):
    """
    直接使用原始数据计算FID
    """
    # 计算均值
    mu_real = np.mean(real_data, axis=0)

    if mean_ is None:
        mu_fake = np.mean(fake_data, axis=0)
    else:
        mu_fake = mean_
    
    # 计算协方差
    sigma_real = np.cov(real_data, rowvar=False)

    if var_ is None:
        sigma_fake = np.cov(fake_data, rowvar=False)
    else:
        sigma_fake = var_
    
    # 计算均值差的平方和
    diff = mu_real - mu_fake
    mean_diff = np.sum(diff * diff)
    
    # 计算协方差项
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    # 计算FID
    fid = mean_diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return float(fid)

from data import Task1Data
from torch.utils.data.dataloader import DataLoader
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

log_dir = 'logs_flow/flow_test'
writer = SummaryWriter(log_dir=log_dir)

train_dataset = Task1Data(is_train=True)
train_loader = DataLoader(train_dataset, batch_size=512,num_workers=8,shuffle=False)

model = ConditionalNormalizingFlow(dim=64, hidden_dim=128, n_layers=2)
model = model.to('cuda')
best_train_loss = 99999.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cc = 0

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
for epoch_counter in range(2000):
    total_train_loss = 0.
    count = 0.

    model.train()
    for step, (x, age, sex) in enumerate(train_loader):
        x = x.to(device, non_blocking=True).float()[..., 0]
        x = F.pad(x, (0, 2))  # 补齐到64维

        age = age.to(device, non_blocking=True).float().view(-1) / 100
        sex = sex.to(device, non_blocking=True).long().view(-1)

        loss = train_step(model, optimizer, x, age, sex, epoch_counter, device)
        
        total_train_loss += len(x) * float(loss)
        count += len(x)

    # scheduler.step()
    total_train_loss /= count
    writer.add_scalar('total_train_loss', total_train_loss, global_step=epoch_counter)
    
    if epoch_counter % 10 == 0:
        print(f"[{epoch_counter}]: train_loss: {total_train_loss:.4f}")

    N_sample = 256
    all_samples = np.zeros((2, 100, N_sample, 64))
    checkpoint_folder = 'checkpoint/flow_model'
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
        
    if epoch_counter % 100 == 0:# and epoch_counter > 0:

        model.eval()
        torch.save(model.state_dict(), f"{checkpoint_folder}/{epoch_counter}.pth")

        for sex in [0,1]:
            for age in range(100):
                # print(f"Conducting... {sex} {age}")
                batch = N_sample
                age_torch = torch.Tensor([age]*batch).to(device,non_blocking=True).float().view(-1) / 100
                sex_torch = torch.Tensor([sex]*batch).to(device,non_blocking=True).long().view(-1)

                samples = model.sample(256, age_torch, sex_torch, device=device)
                all_samples[sex, age] = samples.detach().cpu().numpy()
        fid_errors = []
        for sex in range(2):
            for age in range(15,90):
                mine_data = all_samples[sex,age][:,:62]
                norm_data = train_dataset.ct_feas[(train_dataset.sexs == sex) & (train_dataset.ages == age)][...,0]
                if len(norm_data) <= 1:
                    continue
                fid = calculate_fidV2(mine_data, norm_data)
                fid_errors.append(fid)
        fid_mean_error = np.mean([f for f in fid_errors if f == f])
        print(f"[{epoch_counter}]-fid: {fid_mean_error:.4f}")

        writer.add_scalar('fid_loss', fid_mean_error, global_step=epoch_counter)
