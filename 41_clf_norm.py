import torch
import numpy as np
from data import Task1Data, Task2Data, Task3Data
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
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

use_group = 3

if use_group == 1:
    seeds = [1,21,27,31,37] # asd
    name = 'asd'
    # seeds = range(10)
elif use_group == 2:
    seeds = [0,5,6,8,3] # mci
    name = 'mci'
elif use_group == 3:
    seeds = [9,17,18,19,20] # ad
    name = 'ad'

# dat = np.load("gamlss_norm.npz")
dat = np.load("gpr_norm.npz")
# dat = np.load("loess_norm.npz")


import pandas as pd
df = pd.read_csv("gpr.csv")

use_sex = df[df['group']==use_group]['sex'].values.astype(int)
use_age = df[df['group']==use_group]['age'].values.astype(int)

use_feas = dat[f'{name}'].reshape((len(dat[f'{name}']), -1))
use_dx = dat[f'{name}_dx'].astype(int)
mine_x = use_feas

mine_x[mine_x!=mine_x] = 0
mine_y = use_dx

all_ds = []
for sex in range(2):
    for age in range(100):
        group0 = mine_x[(use_age==age) & (use_sex == sex) & (use_dx == 0)]
        group1 = mine_x[(use_age==age) & (use_sex == sex) & (use_dx == 1)]
        if len(group0) > 1 and len(group1) > 1:
            d = cohens_d(group0, group1)
            all_ds.append(abs(d))
print(f"Cohen'd: {np.mean(all_ds):.4f}")
exit()

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

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
    with open("clf_res/loess_mine.txt", "a") as f:
        f.write(to_print + "\n")

for key in res.keys():
    print(f"{key}: {np.mean(res[key]):.2f} +- {np.std(res[key]):.2f}")