import torch
import torch.nn as nn
import torch.nn.functional as F
class ConditionalCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, condition_dim, mask_type='checkerboard'):
        super().__init__()
        self.dim = dim
        
        # Create alternating mask
        if mask_type == 'checkerboard':
            self.mask = torch.arange(dim) % 2
            self.mask = self.mask.float().view(1, -1)
        
        masked_input_dim = dim // 2 + condition_dim
        
        self.nn = nn.Sequential(
            nn.Linear(dim // 2 + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # 换用GELU激活函数
            nn.Dropout(0.1),  # 添加dropout
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, dim)
        )
    def forward(self, x, condition, reverse=False):
        b, d = x.shape
        mask = self.mask.to(x.device)
        
        # Split input based on mask
        x_masked = x * mask
        x_unmasked = x * (1 - mask)
        
        # Get the masked values
        masked_values = x_masked[mask.bool().expand(b, -1)].view(b, -1)
        
        # Concatenate with condition
        nn_input = torch.cat([masked_values, condition], dim=1)
        s_t = self.nn(nn_input)
        
        # Split into scale and translation
        scale = torch.tanh(s_t[:, :self.dim]) * 0.1  # Constrain scale
        translation = s_t[:, :self.dim]  # Use same size as input
        
        # Reshape scale and translation to match input
        scale = scale * (1 - mask)  # Apply only to unmasked parts
        translation = translation * (1 - mask)  # Apply only to unmasked parts
        
        # Apply transformation
        if not reverse:
            x_transformed = x_unmasked * torch.exp(scale) + translation
            # Compute log determinant
            log_det = torch.sum(scale, dim=1)  # sum over feature dimension
        else:
            x_transformed = (x_unmasked - translation) * torch.exp(-scale)
            log_det = None
            
        return x_masked + x_transformed, log_det if not reverse else x_transformed

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, dim=64, hidden_dim=128, n_layers=4):
        super().__init__()
        self.dim = dim
        
        # Condition encoder (age and gender)
        self.condition_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        condition_dim = hidden_dim // 2
        
        # Stack of coupling layers
        self.layers = nn.ModuleList([
            ConditionalCouplingLayer(dim, hidden_dim, condition_dim, 
                                   mask_type='checkerboard')
            for _ in range(n_layers)
        ])
        
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

def flow_loss(model, x, age, gender):
    z, log_det = model(x, age, gender)
    
    # Standard normal log likelihood
    log_likelihood = -0.5 * torch.sum(z**2, dim=1) - 0.5 * z.size(1) * torch.log(
        torch.tensor(2 * torch.pi).to(z.device))
    
    # Add log determinant of jacobian
    log_likelihood = log_likelihood + log_det
    
    return -log_likelihood.mean()

def train_step(model, optimizer, x, age, sex, epoch, device):
    # 注意这里修复了之前的bug，现在正确传入sex而不是age
    loss = flow_loss(model, x, age, sex)
    
    optimizer.zero_grad()
    loss.backward()
    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    return loss.item(), grad_norm.item()
import os
import numpy as np
model = ConditionalNormalizingFlow(dim=64, hidden_dim=256, n_layers=6)
checkpoint = torch.load('checkpoint/flow_model/1900.pth')
model.load_state_dict(checkpoint)
model = model.to('cuda').eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
name = "flow"
for sex in [0,1]:
    for age in range(100):
        print(f"Conducting... {sex} {age}")
        batch = 256
        age_torch = torch.Tensor([age]*batch).to(device,non_blocking=True).float().view(-1) #/ 100
        sex_torch = torch.Tensor([sex]*batch).to(device,non_blocking=True).long().view(-1)

        samples = model.sample(256, age_torch, sex_torch, device=device)

        to_save = samples.detach().cpu().numpy()
        save_root = f'res/{name}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        np.save(f"{save_root}/sex-{sex}_age-{age}.npy",to_save)
