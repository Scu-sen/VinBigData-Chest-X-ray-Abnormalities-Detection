import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    
    def __init__(self, df, image_path, transform=None):
        self.df = df
        self.image_path = image_path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        labels = torch.from_numpy(
            self.df.loc[idx,np.arange(0,15).astype(str).tolist()].values.astype(float)
        ).float()

        img = cv2.imread(
            self.image_path + '/' + str(self.df.image_id[idx]) + '.png'
        )
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(image=img)['image']
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
            
        return img, labels
    
    
class TestDataset(Dataset):
    
    def __init__(self, df, image_path, transform=None):
        self.df = df
        self.image_path = image_path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        img = cv2.imread(
            self.image_path + '/' + str(self.df.image_id[idx]) + '.png'
        )
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(image=img)['image']
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
            
        return img