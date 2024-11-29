import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import Task1Data
from torch.utils.data.dataloader import DataLoader
from unet1d import Unet1d
from ddpm_conditional import Diffusion
from torch.utils.tensorboard import SummaryWriter
from modules import EMA
import os
import copy
import subprocess


log_dir = "log_mask_bugfix"
checkpoint_dir = "./checkpoint/checkpoint_mask_bugfix"

def try_do(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Error: {e}")

try_do(os.makedirs, log_dir)
try_do(os.makedirs, checkpoint_dir)

writer = SummaryWriter(log_dir=log_dir)
train_dataset = Task1Data(is_train=True)
train_loader = DataLoader(train_dataset, batch_size=128,num_workers=4)
model = Unet1d(64,dim_mults = (1,2,4)) # 

# resume 
# checkpoint = torch.load("best_model.pth", map_location='cpu')
# model.load_state_dict(checkpoint)
#
model.cuda().train()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
diffusion = Diffusion(noise_steps=1000,img_size=64, device=device)
# noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256,
optimizer = torch.optim.AdamW(model.parameters(),lr= 1e-6,weight_decay=0.0001)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

best_train_loss = 99999.

for epoch_counter in range(10000):
    n_steps = 50
    total_train_loss = 0.
    total_val_loss = 0.
    count = 0.

    model.train()
    for step, (x, age, sex) in enumerate(train_loader):
        # padded_tensor = F.pad(tensor, (0, 2))
        x = x.to(device,non_blocking=True).float()[...,0][:,None]
        x = F.pad(x,(0,2))

        # new_x, mask = model.sample_mask(x, x.size(0), 64)
        t = diffusion.sample_timesteps(x.shape[0]).to(device)
        age = age.to(device,non_blocking=True).float().view(-1,1,1)
        sex = sex.to(device,non_blocking=True).float().view(-1,1,1)
        x_t, noise = diffusion.noise_images(x, t)
        predicted_noise, mask = model(x_t, t, age, sex)
        if mask is not None:
            loss = (((noise-x * predicted_noise)**2) * mask).sum() / mask.sum()
        else:
            loss = F.mse_loss(noise, predicted_noise)
        
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)

        optimizer.step()
        total_train_loss += len(x) * float(loss)
        count += len(x)

        ema.step_ema(ema_model, model)

    total_train_loss /= count
    print(f"[{epoch_counter}]: train_loss: {total_train_loss:.4f}")
    writer.add_scalar('total_train_loss', total_train_loss, global_step=epoch_counter)

    if best_train_loss > total_train_loss:
        best_train_loss = total_train_loss
        if epoch_counter > 1000:
            torch.save({
                'model':model.state_dict(),
                'ema_model':ema_model.state_dict(),
                }, os.path.join(f"{checkpoint_dir}/best_model.pth"))

