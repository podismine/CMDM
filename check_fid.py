import numpy as np
from scipy import linalg
import numpy as np
from data import Task1Data, Task2Data, Task3Data

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
def calculate_kl(true_data, fitted_data):
    from scipy.stats import gaussian_kde
    kde_true = gaussian_kde(true_data)
    kde_fitted = gaussian_kde(fitted_data)
    
    # 在一些采样点上计算KL散度
    x_points = np.linspace(min(true_data), max(true_data), 1000)
    p_true = kde_true(x_points)
    p_fitted = kde_fitted(x_points)
    
    eps = 1e-10
    kl_div = np.sum(p_true * np.log((p_true + eps) / (p_fitted + eps))) * (x_points[1] - x_points[0])
    
    return  kl_div

import os
norm_dataset = Task1Data()
abide_dataset = Task2Data()
mci_dataset = Task3Data(no=2)
ad_dataset = Task3Data(no=1)

all_folders = [f for f in os.listdir('res') if 'search_param' in f]
for fold in all_folders:
    fold_path = f'res/{fold}'
    if len(os.listdir(fold_path)) < 200:
        continue
    all_data = np.zeros((2,100,256,62))
    for sex in range(2):
        for age in range(100):
            dat = np.load(f"{fold_path}//sex-{sex}_age-{age}.npy")
            all_data[sex,age] = dat[:,:62]

    fid_errors_norm = []
    fid_errors_asd = []
    fid_errors_mci = []
    fid_errors_ad = []
    for sex in range(2):
        for age in range(15,90):
            mine_data = all_data[sex,age]
            norm_data = norm_dataset.ct_feas[(norm_dataset.sexs == sex) & (norm_dataset.ages == age)][...,0]
            asd_data =  abide_dataset.ct_feas[(abide_dataset.sexs == sex) & (abide_dataset.ages == age) & (abide_dataset.dx == 0)][...,0]
            mci_data =  mci_dataset.ct_feas[(mci_dataset.sexs == sex) & (mci_dataset.ages == age) & (mci_dataset.dx == 0)][...,0]
            ad_data =  ad_dataset.ct_feas[(ad_dataset.sexs == sex) & (ad_dataset.ages == age) & (ad_dataset.dx == 0)][...,0]

            if len(norm_data) > 2:
                fid = calculate_fid(mine_data, norm_data)
                fid_errors_norm.append(fid)
            if len(asd_data) > 2:
                fid = calculate_fid(mine_data, asd_data)
                fid_errors_asd.append(fid)
            if len(mci_data) > 2:
                fid = calculate_fid(mine_data, mci_data)
                fid_errors_mci.append(fid)
            if len(ad_data) > 2:
                fid = calculate_fid(mine_data, ad_data)
                fid_errors_ad.append(fid)
    print(f"{fold_path}-fid-norm: {np.mean([f for f in fid_errors_norm if f == f]):.4f} fid-asd: {np.mean([f for f in fid_errors_asd if f == f]):.4f} fid-mci: {np.mean([f for f in fid_errors_mci if f == f]):.4f} fid-ad: {np.mean([f for f in fid_errors_ad if f == f]):.4f}")