import torch
import numpy as np
import os
from ddp import MaskedDDPM1D
# model = MaskedDDPM1D(input_dim=64, hidden_dims= [128, 256, 512], condition_dim= 256).cuda()
# model = MaskedDDPM1D(input_dim=64, hidden_dims= [128, 256, 512], condition_dim= 64).cuda()
# model = MaskedDDPM1D(input_dim=64, hidden_dims= [256, 512, 1024], condition_dim= 64).cuda()
# model = MaskedDDPM1D(input_dim=64, hidden_dims= [128,256, 512, 1024], condition_dim= 64).cuda()
model = MaskedDDPM1D(input_dim=64, hidden_dims= [256, 512], condition_dim= 64).cuda()

# model = MaskedDDPM1D(input_dim=64).cuda()
name = 'mask_bugfix15_2layer_3e4_shortrain'
checkpoint_dir = f"checkpoint/checkpoint_{name}/"
checkpoint = torch.load(f"{checkpoint_dir}/last_model.pth", map_location='cpu',weights_only=True)
model.load_state_dict(checkpoint)
model.cuda().eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for sex in [0,1]:
    for age in range(100):
        print(f"Conducting... {sex} {age}")
        batch = 256
        age_torch = torch.Tensor([age]*batch).to(device,non_blocking=True).float().view(-1,1) / 100
        sex_torch = torch.Tensor([sex]*batch).to(device,non_blocking=True).long().view(-1)

        mask = torch.ones((batch, 64)).to(device,non_blocking=True)
        samples = model.sample(mask, age_torch, sex_torch, device = device)

        to_save = samples.detach().cpu().numpy()
        # print(to_save[0])
        # exit()
        save_root = f'res/{name}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        np.save(f"{save_root}/sex-{sex}_age-{age}.npy",to_save)

        del mask, samples