import torch
import numpy as np
from data import Task1Data, Task2Data, Task3Data
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from scipy import stats

def cohens_d(group1, group2, pooled=True):
    """
    Calculate Cohen's d effect size between two groups.
    
    Parameters:
    -----------
    group1 : array-like
        First group of measurements
    group2 : array-like
        Second group of measurements
    pooled : bool, default=True
        Whether to use pooled standard deviation (True) or group2's std (False)
    
    Returns:
    --------
    d : float
        Cohen's d effect size
    
    Notes:
    ------
    Formula for pooled standard deviation:
    s_pooled = sqrt(((n₁ - 1)s₁² + (n₂ - 1)s₂²) / (n₁ + n₂ - 2))
    
    Cohen's d = (M₁ - M₂) / s_pooled
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Calculate means
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    # Calculate pooled standard deviation
    if pooled:
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    else:
        pooled_std = np.sqrt(var2)  # 使用第二组的标准差
    
    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return d

norm_dataset = Task1Data()
abide_dataset = Task2Data()
mci_dataset = Task3Data(no=2)
ad_dataset = Task3Data(no=1)

check_bins = 20
name = 'res/search_param0.002_mom0.05/'
name = 'res/search_param0.01_mom0.1/'
statis_dist = np.zeros((2,100,62, check_bins + 1))
statis_std = np.zeros((2,100,62, 1))
statis_mean = np.zeros((2,100,62, 1))
print("Loading data...")
for sex in range(2):
    for age in range(100):
        dat = np.load(f"{name}//large_sex-{sex}_age-{age}.npy")
        for bins in range(1,check_bins + 1):
            statis_dist[sex,age,:,bins-1] = np.quantile(dat[:,:62],bins/check_bins, axis=0)
        statis_std[sex,age,:,0] = dat.std(0).reshape(-1)[:62]
        statis_dist[sex,age,:,-1] = dat.mean(0).reshape(-1)[:62]
        statis_mean[sex,age,:,-1] = dat.mean(0).reshape(-1)[:62]
print("Loading finished...")

use_data = ad_dataset

use_feas = use_data.ct_feas[:,:,0]#[...,None].repeat(check_bins+1,axis=2) # N, 62
use_age = use_data.ages
use_sex = use_data.sexs
use_dx = use_data.dx

mine_feas = np.zeros_like(use_feas)
for i in range(len(mine_feas)):
    sex_ = use_sex[i].astype(int)
    age_ = use_age[i].astype(int)
    mean_ = statis_mean[sex_, age_,:].reshape((-1)) # 62,
    std_ = statis_std[sex_, age_].reshape((-1))# 62, 1
    mine_feas[i] = (use_feas[i] - mean_) / std_

# mine_feas = (use_feas - statis_dist[use_sex.astype(int),use_age.astype(int)] ) / statis_std[use_sex.astype(int),use_age.astype(int)]
# mine_x = mine_feas.reshape(-1, 62* (check_bins+1))
# raw_x = use_feas.reshape(-1, 62* (check_bins+1))
# mine_x = np.concatenate([mine_x, raw_x],-1)
mine_x = mine_feas#[...,-1].reshape(len(mine_feas),-1)
# mine_x = raw_x
mine_y = use_dx
print(mine_x.shape, mine_y.shape)

all_ds = []
for sex in range(2):
    for age in range(100):
        group0 = mine_feas[(use_age==age) & (use_sex == sex) & (use_dx == 0)]
        group1 = mine_feas[(use_age==age) & (use_sex == sex) & (use_dx == 1)]
        if len(group0) > 1 and len(group1) > 1:
            d = cohens_d(group0, group1)
            all_ds.append(abs(d))
print(f"Cohen'd: {np.mean(all_ds):.4f}")
exit()
seeds = [1,21,27,31,37] # asd
# seeds = [18,6,5,8,38] # mci
# seeds = [9,17,18,19,20] # ad

# seeds = range(40)
res = {
    'acc':[],
    'sen':[],
    'spe':[],
    'auc':[]
}
for seed in seeds:
    split = StratifiedShuffleSplit(n_splits=1, test_size=int(0.2 * len(mine_x)), random_state=seed)
    for train_index, test_index in split.split(mine_x, mine_y):
        train_x, train_y = mine_x[train_index], mine_y[train_index]
        test_x, test_y = mine_x[test_index], mine_y[test_index]

    norm = StandardScaler()
    train_x = norm.fit_transform(train_x)
    test_x = norm.transform(test_x)
    clf = SVC(probability=True, random_state=42)
    clf.fit(train_x, train_y)

    y_pred = clf.predict(test_x)
    y_pred_prob = clf.predict_proba(test_x)[:, 1]
    tn, fp, fn, tp = confusion_matrix(test_y, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) * 100  # 也叫召回率 recall
    specificity = tn / (tn + fp) * 100
    accuracy = accuracy_score(test_y, y_pred) * 100
    
    fpr, tpr, _ = roc_curve(test_y, y_pred_prob)
    roc_auc = auc(fpr, tpr) * 100
    for key,val in zip(['acc','sen','spe','auc'],[accuracy,sensitivity,specificity,roc_auc]):
        res[key].append(val)
    to_print = f"seed: {seed}, sensitivity: {sensitivity:.2f}, specificity: {specificity:.2f}, accuracy: {accuracy:.2f}, roc_auc: {roc_auc:.2f}"
    print(to_print)
    with open("clf_res/mine.txt", "a") as f:
        f.write(to_print + "\n")

for key in res.keys():
    print(f"{key}: {np.mean(res[key]):.2f} +- {np.std(res[key]):.2f}")