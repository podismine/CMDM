import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import Task1Data, Task3Data
from torch.utils.data.dataloader import DataLoader
from unet1d import Unet1d
from ddpm_conditional import Diffusion
from torch.utils.tensorboard import SummaryWriter
import os

model = Unet1d(64,dim_mults = (1,2,4)) # 
# checkpoint_dir = "./checkpoint/checkpoint_maskDDPM"
# checkpoint_dir = "./checkpoint/checkpoint_mask_bugfix"
# checkpoint_dir = "./checkpoint/checkpoint_ft_mask_bugfix"
checkpoint_dir = "checkpoint/checkpoint_rerun"
checkpoint = torch.load(f"{checkpoint_dir}/best_model.pth", map_location='cpu')
model.load_state_dict(checkpoint['ema_model'])
model.cuda().eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
diffusion = Diffusion(noise_steps=1000,img_size=64, device=device)

for sex in [0,1]:
    for age in ([i for i in range(20)] + [i for i in range(86,100)]):
        print(f"Conducting... {sex} {age}")
        batch = 512
        age_torch = torch.Tensor([age]*batch).to(device,non_blocking=True).float().view(-1,1,1)
        sex_torch = torch.Tensor([sex]*batch).to(device,non_blocking=True).float().view(-1,1,1)
        sampled_images = diffusion.sample(model,age_torch,sex_torch)

        to_save = sampled_images.detach().cpu().numpy()
        np.save(f"mask_res/sex-{sex}_age-{age}.npy",to_save)