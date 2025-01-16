#coding:utf8
import os
from torch.utils import data
import numpy as np
import nibabel as nib
import random
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
import numpy as np
from sklearn.utils import shuffle


warnings.filterwarnings("ignore")

def map_sex(val):
    sex_match = {
        'M':['Male','m','M',1,' m ',1],
        'F':['Female','f','F',2,' f ',0]
    }
    if val in sex_match['M']:
        return 0
    elif val in sex_match['F']:
        return 1
    else:
        raise KeyError(f"invalid sex value: {val}")

def filter_age(arr, is_test= False):
    ct_feas, gmv_feas, ages, sexs = arr['ct'].astype(float), arr['gmv'].astype(float), arr['age'].astype(int), arr['sex']
    feas = np.stack([ct_feas, gmv_feas],-1)

    use_mask = (ages < 100) & (ages > 0)
    use_sex = np.array([map_sex(f) for f in sexs[use_mask]]).astype(float)
    if is_test is True:
        dxs = arr['dx'][use_mask]
        return feas[use_mask],ages[use_mask],use_sex,dxs

    return feas[use_mask],ages[use_mask],use_sex

def normalize(arr):
    mean_feas = arr.mean(0)
    std_feas = arr.std(0)
    min_feas = arr.min(0)
    max_feas = arr.max(0)

    # np.savez("./norm_vals.npz", mean = mean_feas, std = std_feas, min = min_feas, max = max_feas)
    # norm_feas = (arr-min_feas)/(max_feas-min_feas)
    # norm_feas = (arr-mean_feas)/std_feas
    norm_feas = arr.copy()
    norm_feas[...,0] = norm_feas[...,0]/5
    return norm_feas

def normalize_inference(arr):
    # norm = np.load("./norm_vals.npz")
    # mean_feas = norm['mean']
    # std_feas = norm['std']
    # min_feas = norm['min']
    # max_feas = norm['max']
    # norm_feas = (arr-mean_feas)/std_feas
    # norm_feas = (arr-min_feas)/(max_feas-min_feas)
    # norm_feas[norm_feas<0] = 0
    # norm_feas[norm_feas>1] = 1
    norm_feas = arr.copy()
    norm_feas[...,0] = norm_feas[...,0]/5
    return norm_feas

class Task1Data(data.Dataset):

    def __init__(self, mask_ratio = 0.1,is_train=False):
        file_list = ['data/hcpa.npz', \
                     "data/fcon.npz", \
                     "data/camcan.npz", \
                     ]

        self.ct_feas = []
        self.ages = []
        self.sexs = []

        for file in file_list:
            dt = np.load(file,allow_pickle = True)
            dt_ct, dt_age, dt_sex = filter_age(dt)
            print(dt_ct.shape, dt_age.shape, dt_sex.shape)
            self.ct_feas.append(dt_ct)
            self.ages.append(dt_age)
            self.sexs.append(dt_sex)
        # self.ct_feas = np.concatenate(self.ct_feas)
        self.ct_feas = normalize(np.concatenate(self.ct_feas))
        self.ages = np.concatenate(self.ages)
        self.sexs = np.concatenate(self.sexs)

        all_index = np.arange(len(self.ct_feas))
        all_index = shuffle(all_index,random_state=42)
        if is_train is True:
            use_index = all_index#[:int(len(self.ct_feas) * 0.8)]
            self.ct_feas, self.ages, self.sexs = self.ct_feas[use_index], self.ages[use_index], self.sexs[use_index]
        else:
            use_index = all_index#[int(len(self.ct_feas)* 0.8):]
            self.ct_feas, self.ages, self.sexs = self.ct_feas[use_index], self.ages[use_index], self.sexs[use_index]

        print(f"Finding files: {len(self.ct_feas)}/{len(self.ages)}/{len(self.sexs)}")

    def __getitem__(self,index):
        ct = np.array(self.ct_feas[index]).astype(float)#.reshape(-1,2)
        age = np.array(self.ages[index]).reshape(-1,1)
        sex = np.array(self.sexs[index]).reshape(-1,1)
        return ct, age, sex

    def __len__(self):
        return len(self.ct_feas)

class Task2Data(data.Dataset):

    def __init__(self, mask_ratio = 0.1):
        dt = np.load("data/abide1.npz",allow_pickle = True)
        dt_ct, dt_age, dt_sex,dt_dx = filter_age(dt,is_test = True)

        self.ct_feas = normalize_inference(dt_ct)
        self.ages = dt_age
        self.sexs = dt_sex
        self.dx = np.array([1 if f ==1 else 0 for f in dt_dx])#raw 1 asd 2 nc
        print(f"Finding files: {len(self.ct_feas)}/{len(self.ages)}/{len(self.sexs)}")

    def __getitem__(self,index):
        ct = np.array(self.ct_feas[index]).astype(float)#.reshape(len(self.ct_feas[index]),-1)
        age = np.array(self.ages[index])
        sex = self.sexs[index]
        dx = self.dx[index]
        return ct, age, sex, dx

    def __len__(self):
        return len(self.ct_feas)
    
def filter_label(no, lbl, *args):
    mask = lbl!=no
    min_label = np.min(lbl)
    max_label = np.max(lbl)

    lbl[lbl==min_label] = 0
    lbl[lbl==max_label] = 1
    ret = [f[mask] for f  in args]
    return lbl[mask],*ret

class Task3Data(data.Dataset):

    def __init__(self, no=1):
        dt = np.load("data/adni.npz",allow_pickle = True)
        dt_ct, dt_age, dt_sex,dt_dx = filter_age(dt,is_test = True)

        self.ct_feas = normalize_inference(dt_ct)
        self.ages = dt_age
        self.sexs = dt_sex
        self.dx = dt_dx

        self.dx, self.ct_feas, self.ages,self.sexs = filter_label(no, self.dx, self.ct_feas, self.ages,self.sexs)
        print(f"Finding files: {len(self.ct_feas)}/{len(self.ages)}/{len(self.sexs)}")
        print(f"{len(self.dx[self.dx==1])}/{len(self.dx[self.dx==0])}")
    def __getitem__(self,index):
        ct = np.array(self.ct_feas[index]).astype(float)#.reshape(len(self.ct_feas[index]),-1)
        age = np.array(self.ages[index])
        sex = self.sexs[index]
        dx = self.dx[index]
        return ct, age, sex, dx

    def __len__(self):
        return len(self.ct_feas)
    

if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
        
    train_dataset = Task1Data()
    train_loader = DataLoader(train_dataset, 32,num_workers=0)
