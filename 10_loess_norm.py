import numpy as np
import pandas as pd
from pynm.pynm import PyNM
import seaborn as sns

import numpy as np
from data import Task1Data, Task2Data, Task3Data
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

def calculate_kl(true_data, fitted_data):
    from scipy.stats import gaussian_kde
    kde_true = gaussian_kde(true_data.ravel())
    kde_fitted = gaussian_kde(fitted_data.ravel())
    
    # 在一些采样点上计算KL散度
    x_points = np.linspace(min(true_data), max(true_data), 100)
    p_true = kde_true(x_points)
    p_fitted = kde_fitted(x_points)
    
    eps = 1e-10
    kl_div = np.sum(p_true * np.log((p_true + eps) / (p_fitted + eps))) * (x_points[1] - x_points[0])
    
    return  kl_div


norm_dataset = Task1Data()
abide_dataset = Task2Data()
mci_dataset = Task3Data(no=2)
ad_dataset = Task3Data(no=1)

pred_data_norm = np.zeros((len(norm_dataset.ct_feas), 62))
pred_data_asd = np.zeros((len(abide_dataset.ct_feas), 62))
pred_data_mci = np.zeros((len(mci_dataset.ct_feas), 62))
pred_data_ad = np.zeros((len(ad_dataset.ct_feas), 62))
# pred_data=[]
for roi in range(62):
    df = pd.DataFrame()
    df['fea'] = np.concatenate([norm_dataset.ct_feas[...,0][:,roi], \
                                abide_dataset.ct_feas[...,0][:,roi], \
                                    mci_dataset.ct_feas[...,0][:,roi], \
                                        ad_dataset.ct_feas[...,0][:,roi]])
    df['age'] = np.concatenate([norm_dataset.ages, abide_dataset.ages, mci_dataset.ages, ad_dataset.ages])
    df['sex'] = np.concatenate([norm_dataset.sexs, abide_dataset.sexs, mci_dataset.sexs, ad_dataset.sexs]) #sexs
    df['male'] = 1 -df['sex']
    df['group'] =   [0] * len(norm_dataset.ct_feas) + \
                        [1] * len(abide_dataset.ct_feas) + \
                            [2] * len(mci_dataset.ct_feas) + \
                                [3] * len(ad_dataset.ct_feas)
    df['use_dx'] = np.concatenate([np.zeros_like((norm_dataset.sexs)), abide_dataset.dx, mci_dataset.dx, ad_dataset.dx])

    m = PyNM(df,'fea','group',confounds = ['age','sex'],bin_spacing=1,bin_width=7)

    m.loess_normative_model()
    # pred = m.data['LOESS_pred'].values

    to_save = 'LOESS_pred' # 'LOESS_z'
    pred_data_norm[:,roi] = m.data[m.data['group']==0][to_save].values  #pred#.reshape(-1)
    pred_data_asd[:,roi] = m.data[m.data['group']==1][to_save].values  #pred#.reshape(-1)
    pred_data_mci[:,roi] = m.data[m.data['group']==2][to_save].values  #pred#.reshape(-1)
    pred_data_ad[:,roi] = m.data[m.data['group']==3][to_save].values  #pred#.reshape(-1)
# m.data.to_csv("loess.csv",index=None)
np.savez("loess_norm.npz",asd=pred_data_asd, mci=pred_data_mci, ad=pred_data_ad, norm=pred_data_norm,\
                         asd_dx = abide_dataset.dx, mci_dx = mci_dataset.dx, ad_dx = ad_dataset.dx,\
                        asd_raw = abide_dataset.ct_feas[...,0], mci_raw = mci_dataset.ct_feas[...,0], ad_raw = ad_dataset.ct_feas[...,0])
fid_errors_norm = []
fid_errors_asd = []
fid_errors_mci = []
fid_errors_ad = []
for sex in range(2):
    for age in range(15,90):
        pred_norm_data = pred_data_norm[(norm_dataset.sexs == sex) & (norm_dataset.ages == age)]
        gt_norm_data = norm_dataset.ct_feas[(norm_dataset.sexs == sex) & (norm_dataset.ages == age)][...,0]

        pred_asd_data = pred_data_asd[(abide_dataset.sexs == sex) & (abide_dataset.ages == age) & (abide_dataset.dx == 0)]
        gt_asd_data = abide_dataset.ct_feas[(abide_dataset.sexs == sex) & (abide_dataset.ages == age) & (abide_dataset.dx == 0)][...,0]

        pred_mci_data = pred_data_mci[(mci_dataset.sexs == sex) & (mci_dataset.ages == age)  & (mci_dataset.dx == 0) ]
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
print(f"LOESS-fid-norm: {np.mean([f for f in fid_errors_norm if f == f]):.4f} fid-asd: {np.mean([f for f in fid_errors_asd if f == f]):.4f} fid-mci: {np.mean([f for f in fid_errors_mci if f == f]):.4f} fid-ad: {np.mean([f for f in fid_errors_ad if f == f]):.4f}")