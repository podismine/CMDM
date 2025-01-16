from cvae import cVAE
import torch
from data import Task1Data
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

h_dim = [128,256]
# h_dim = [64,128]
z_dim = 8
input_dim = 64

train_dataset = Task1Data(is_train=True)
train_loader = DataLoader(train_dataset, batch_size=512,num_workers=8,shuffle=False)

DEVICE = torch.device("cuda")
model = cVAE(input_dim=input_dim, hidden_dim=h_dim, latent_dim=z_dim, c_dim=2, learning_rate=0.0001, non_linear=True)
model.to(DEVICE)

model.optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-4)
model.optimizer2 = torch.optim.Adam(list(model.discriminator.parameters()), lr=1e-4)
model.optimizer3 = torch.optim.Adam(list(model.encoder.parameters()), lr=1e-4)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="fmri tokenizer")

    parser.add_argument('-a', "--alpha", type=float,default=0)
    parser.add_argument('-g', "--gamma", type=float,default=1)
    parser.add_argument('-r', "--rval", type=float,default=0.1)

    args = parser.parse_args()

    return args

args = parse_args()

for epoch_counter in range(200):
    n_steps = 50
    total_train_loss = 0.
    total_val_loss = 0.
    count = 0.

    model.train()
    for step, (x, age, sex) in enumerate(train_loader):
        x = x.to(DEVICE,non_blocking=True).float()[...,0]#[:,None]
        x = F.pad(x,(0,2))

        age = age.to(DEVICE,non_blocking=True).float().view(-1, 1) / 100
        sex = sex.to(DEVICE,non_blocking=True).float().view(-1, 1)

        cov = torch.cat([age,sex],-1)

        fwd_rtn = model.forward(x, cov)
        loss = model.loss_function(x, fwd_rtn)
        model.optimizer.zero_grad()
        loss['total'].backward()
        model.optimizer.step() 

        fwd_rtn2 = model.forward2(x, cov, z_dim)
        loss2 = model.loss_function2(x = x, fwd_rtn = fwd_rtn2, alpha_focal=args.alpha, gamma_focal=args.gamma, lambda_reg=args.rval, logits=True, reduction='mean')
        model.optimizer2.zero_grad()
        loss2['dc_loss'].backward()
        model.optimizer2.step() 

        fwd_rtn3 = model.forward3(x, cov)
        loss3 = model.loss_function3(x, fwd_rtn3)
        model.optimizer3.zero_grad()
        loss3['gen_loss'].backward()
        model.optimizer3.step() 
        
        to_print = 'Train Epoch:' + str(epoch_counter) + ' ' + 'Train batch: ' + str(step) + ' '+ ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
        print(to_print)        
torch.save(model, f"checkpoint/FAVAE_a{args.alpha}_g{args.gamma}_r{args.rval}.pkl")
