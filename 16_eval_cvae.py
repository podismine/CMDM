import torch
import numpy as np
import torch.nn.functional as F
from data import Task1Data, Task2Data, Task3Data
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch

norm_dataset = Task1Data()
abide_dataset = Task2Data()
mci_dataset = Task3Data(no=2)
ad_dataset = Task3Data(no=1)

norm_loader = DataLoader(norm_dataset, batch_size=64,num_workers=0,shuffle=False)
abide_loader = DataLoader(abide_dataset, batch_size=64,num_workers=0,shuffle=False)
mci_loader = DataLoader(mci_dataset, batch_size=64,num_workers=0,shuffle=False)
ad_loader = DataLoader(ad_dataset, batch_size=64,num_workers=0,shuffle=False)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="fmri tokenizer")

    parser.add_argument('-a', "--alpha", type=float,default=0.01)
    parser.add_argument('-g', "--gamma", type=float,default=20.0)
    parser.add_argument('-r', "--rval", type=float,default=0.001)

    args = parser.parse_args()

    return args

args = parse_args()

model = torch.load(f"checkpoint/FAVAE_a{args.alpha}_g{args.gamma}_r{args.rval}.pkl") # a0.01_g20.0_r0.001
model.cuda().eval()
DEVICE = torch.device("cuda")

from scipy import linalg
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
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
    mu_real = np.mean(real_data, axis=0)

    if mean_ is None:
        mu_fake = np.mean(fake_data, axis=0)
    else:
        mu_fake = mean_
    
    sigma_real = np.cov(real_data, rowvar=False)

    if var_ is None:
        sigma_fake = np.cov(fake_data, rowvar=False)
    else:
        sigma_fake = var_
    
    diff = mu_real - mu_fake
    mean_diff = np.sum(diff * diff)
    
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = mean_diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return float(fid)

repeat = 1
pred_data_norm = np.zeros((len(norm_dataset.ct_feas) * repeat, 62))
pred_norm_sex = np.zeros((len(norm_dataset.ct_feas) * repeat, 1))
pred_norm_age = np.zeros((len(norm_dataset.ct_feas) * repeat, 1))

pred_data_asd = np.zeros((len(abide_dataset.ct_feas) * repeat, 62))
pred_data_asd_err = np.zeros((len(abide_dataset.ct_feas) * repeat, 62))
pred_asd_sex = np.zeros((len(abide_dataset.ct_feas) * repeat, 1))
pred_asd_age = np.zeros((len(abide_dataset.ct_feas) * repeat, 1))
pred_asd_dx = np.zeros((len(abide_dataset.ct_feas) * repeat, 1))

pred_data_mci = np.zeros((len(mci_dataset.ct_feas) * repeat, 62))
pred_data_mci_err = np.zeros((len(mci_dataset.ct_feas) * repeat, 62))
pred_mci_sex = np.zeros((len(mci_dataset.ct_feas) * repeat, 1))
pred_mci_age = np.zeros((len(mci_dataset.ct_feas) * repeat, 1))
pred_mci_dx = np.zeros((len(mci_dataset.ct_feas) * repeat, 1))

pred_data_ad = np.zeros((len(ad_dataset.ct_feas), 62))
pred_data_ad_err = np.zeros((len(ad_dataset.ct_feas), 62))
pred_ad_sex = np.zeros((len(ad_dataset.ct_feas) * repeat, 1))
pred_ad_age = np.zeros((len(ad_dataset.ct_feas) * repeat, 1))
pred_ad_dx = np.zeros((len(ad_dataset.ct_feas), 1))

for step, (x, age, sex) in enumerate(norm_loader):
    x = x.to(DEVICE,non_blocking=True).float()[...,0]#[:,None]
    x = F.pad(x,(0,2))
    age = age.to(DEVICE,non_blocking=True).float().view(-1, 1) / 100
    sex = sex.to(DEVICE,non_blocking=True).float().view(-1, 1)

    cov = torch.cat([age,sex],-1)
    in_x = x.repeat(repeat,1)
    in_conv = cov.repeat(repeat,1)
    with torch.no_grad():
        test_prediction = model.pred_recon(in_x, in_conv, DEVICE)[:,:62]
        
    pred_data_norm[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction)) ] = test_prediction.detach().cpu().numpy()
    pred_norm_sex[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] = sex.repeat(repeat,1).detach().cpu().numpy().astype(int)
    pred_norm_age[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] = (100 * age.repeat(repeat,1).detach().cpu().numpy()).astype(int)


for step, (x, age, sex, dx) in enumerate(abide_loader):
    x = x.to(DEVICE,non_blocking=True).float()[...,0]#[:,None]
    x = F.pad(x,(0,2))
    age = age.to(DEVICE,non_blocking=True).float().view(-1, 1) / 100
    sex = sex.to(DEVICE,non_blocking=True).float().view(-1, 1)

    cov = torch.cat([age,sex],-1)
    in_x = x.repeat(repeat,1)
    in_conv = cov.repeat(repeat,1)
    with torch.no_grad():
        test_prediction = model.pred_recon(in_x, in_conv, DEVICE)[:,:62]
        
    pred_data_asd[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction)) ] = test_prediction.detach().cpu().numpy()
    pred_data_asd_err[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction)) ] = (test_prediction - x[:,:62]).detach().cpu().numpy()
    pred_asd_sex[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] = sex.repeat(repeat,1).detach().cpu().numpy().astype(int)
    pred_asd_age[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] = (100 * age.repeat(repeat,1).detach().cpu().numpy()).astype(int)
    pred_asd_dx[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] =dx.numpy().astype(int)[:,None]

for step, (x, age, sex, dx) in enumerate(mci_loader):
    x = x.to(DEVICE,non_blocking=True).float()[...,0]#[:,None]
    x = F.pad(x,(0,2))
    age = age.to(DEVICE,non_blocking=True).float().view(-1, 1) / 100
    sex = sex.to(DEVICE,non_blocking=True).float().view(-1, 1)

    cov = torch.cat([age,sex],-1)
    in_x = x.repeat(repeat,1)
    in_conv = cov.repeat(repeat,1)
    with torch.no_grad():
        test_prediction = model.pred_recon(in_x, in_conv, DEVICE)[:,:62]
        
    pred_data_mci[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction)) ] = test_prediction.detach().cpu().numpy()
    pred_data_mci_err[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction)) ] = (test_prediction - x[:,:62]).detach().cpu().numpy()
    pred_mci_sex[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] = sex.repeat(repeat,1).detach().cpu().numpy().astype(int)
    pred_mci_age[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] = (100 * age.repeat(repeat,1).detach().cpu().numpy()).astype(int)
    pred_mci_dx[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] =dx.numpy().astype(int)[:,None]

for step, (x, age, sex, dx) in enumerate(ad_loader):
    x = x.to(DEVICE,non_blocking=True).float()[...,0]#[:,None]
    x = F.pad(x,(0,2))
    age = age.to(DEVICE,non_blocking=True).float().view(-1, 1) / 100
    sex = sex.to(DEVICE,non_blocking=True).float().view(-1, 1)

    cov = torch.cat([age,sex],-1)

    with torch.no_grad():
        test_prediction = model.pred_recon(x, cov, DEVICE)[:,:62]
    pred_data_ad[step * len(x): (step * len(x) + len(x)) ] = test_prediction.detach().cpu().numpy()
    pred_data_ad_err[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction)) ] = (test_prediction - x[:,:62]).detach().cpu().numpy()
    pred_ad_sex[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] = sex.repeat(repeat,1).detach().cpu().numpy().astype(int)
    pred_ad_age[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] = (100 * age.repeat(repeat,1).detach().cpu().numpy()).astype(int)
    pred_ad_dx[step * len(test_prediction): (step * len(test_prediction)+ len(test_prediction))] =dx.numpy().astype(int)[:,None]

np.savez("favae_norm.npz",asd=pred_data_asd_err, mci=pred_data_mci_err, ad=pred_data_ad_err, norm=pred_data_norm,\
                         asd_dx = pred_asd_dx, mci_dx = pred_mci_dx, ad_dx = pred_ad_dx,\
                            asd_sex =pred_asd_sex, mci_sex = pred_mci_sex, ad_sex = pred_ad_sex,\
                            asd_age =pred_asd_age, mci_age = pred_mci_age, ad_age = pred_ad_age,\
                                 )

fid_errors_norm = []
fid_errors_asd = []
fid_errors_mci = []
fid_errors_ad = []
for sex in range(2):
    for age in range(15,90):
        pred_norm_data = pred_data_norm[(pred_norm_sex[:,0] == sex) & (pred_norm_age[:,0] == age),:]
        gt_norm_data = norm_dataset.ct_feas[(norm_dataset.sexs == sex) & (norm_dataset.ages == age)][...,0]

        pred_asd_data = pred_data_asd[(pred_asd_sex[:,0] == sex) & (pred_asd_age[:,0] == age),:]
        gt_asd_data = abide_dataset.ct_feas[(abide_dataset.sexs == sex) & (abide_dataset.ages == age) & (abide_dataset.dx == 0)][...,0]

        pred_mci_data = pred_data_mci[(pred_mci_sex[:,0] == sex) & (pred_mci_age[:,0] == age),:]
        gt_mci_data = mci_dataset.ct_feas[(mci_dataset.sexs == sex) & (mci_dataset.ages == age) & (mci_dataset.dx == 0) ][...,0]

        pred_ad_data = pred_data_ad[(ad_dataset.sexs == sex) & (ad_dataset.ages == age)  & (ad_dataset.dx == 0)]
        gt_ad_data = ad_dataset.ct_feas[(ad_dataset.sexs == sex) & (ad_dataset.ages == age) & (ad_dataset.dx == 0) ][...,0]

        if len(pred_norm_data) > 2 and len(gt_norm_data) > 2:
            fid = calculate_fidV2(pred_norm_data, gt_norm_data)
            fid_errors_norm.append(fid)

        if len(pred_asd_data) > 2 and len(gt_asd_data) > 2:
            fid = calculate_fidV2(pred_asd_data, gt_asd_data)
            fid_errors_asd.append(fid)

        if len(pred_mci_data) > 2 and len(gt_mci_data) > 2:
            fid = calculate_fidV2(pred_mci_data, gt_mci_data)
            fid_errors_mci.append(fid)

        if len(pred_ad_data) > 2 and len(gt_ad_data) > 2:
            fid = calculate_fidV2(pred_ad_data, gt_ad_data)
            fid_errors_ad.append(fid)

# print(f"LOESS-fid-norm: {np.mean([f for f in fid_errors_norm if f == f]):.4f} fid-asd: {np.mean([f for f in fid_errors_asd if f == f]):.4f} ")
to_print = f"ACVAE-fid-norm: {np.mean([f for f in fid_errors_norm if f == f]):.4f} fid-asd: {np.mean([f for f in fid_errors_asd if f == f]):.4f} fid-mci: {np.mean([f for f in fid_errors_mci if f == f]):.4f} fid-ad: {np.mean([f for f in fid_errors_ad if f == f]):.4f}"
print(to_print)
with open("res_faae.txt", 'a') as f:
    f.write(f"FAVAE_a{args.alpha}_g{args.gamma}_r{args.rval}: "+to_print +'\n')