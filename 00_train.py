import torch, os
from data import Task1Data
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ddp import MaskedDDPM1D
import torch.nn.functional as F


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
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)

    optimizer.step()
    
    return loss.item()

if __name__ == "__main__":
    name = "mask_bugfix15_2layer_3e4_shortrain"
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

    # model = MaskedDDPM1D(input_dim=64, hidden_dims= [128, 256, 512], condition_dim= 64).cuda() # 11
    model = MaskedDDPM1D(input_dim=64, hidden_dims= [256, 512], condition_dim= 64).cuda()

    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    # optimizer = torch.optim.AdamW(model.parameters())
    # optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    # loss = train_step(model, optimizer, x, mask, device='cpu')
    # samples = model.sample(x, mask)

    train_dataset = Task1Data(is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=512,num_workers=8)

    best_train_loss = 99999.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask_ratio = 0.2
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

            loss = train_step(model, optimizer, x, mask, age, sex, device=device)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)

            total_train_loss += len(x) * float(loss)
            count += len(x)
        
        total_train_loss /= count
        print(f"[{epoch_counter}]: train_loss: {total_train_loss:.4f}")
        writer.add_scalar('total_train_loss', total_train_loss, global_step=epoch_counter)

        if epoch_counter % 100 == 0:
            torch.save(model.state_dict(), os.path.join(f"{checkpoint_dir}/{epoch_counter}_model.pth"))

        if best_train_loss > total_train_loss:
            best_train_loss = total_train_loss
            torch.save(model.state_dict(), os.path.join(f"{checkpoint_dir}/best_model.pth"))

    torch.save(model.state_dict(), os.path.join(f"{checkpoint_dir}/last_model.pth"))
