import torch
import numpy as np
from data import Task1Data, Task2Data, Task3Data
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt


abide_dataset = Task2Data()
mci_dataset = Task3Data(no=2)
ad_dataset = Task3Data(no=1)

use_data = abide_dataset
use_feas = use_data.ct_feas[:,:,0]
use_age = use_data.ages
use_sex = use_data.sexs
use_dx = use_data.dx

mine_x = use_feas.reshape((len(use_feas), -1))
mine_y = use_dx

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler



seeds = [1,21,27,31,37] # asd
# seeds = [0,5,6,8,3] # mci
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
    with open("clf_res/baseline.txt", "a") as f:
        f.write(to_print + "\n")

for key in res.keys():
    print(f"{key}: {np.mean(res[key]):.2f} +- {np.std(res[key]):.2f}")