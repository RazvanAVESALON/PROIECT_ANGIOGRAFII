from __future__ import annotations
from tkinter import Y, image_names
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import random
import yaml
import cv2
import json
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchmetrics 

class AngioClass(torch.utils.data.Dataset):
    def __init__(self, dataset_df):
        self.dataset_df = dataset_df.reset_index(drop=True)
      
    
    def __len__(self):
      
       return len(self.dataset_df) 

    def __getitem__(self, idx):
        """Returneaza un tuple (input, target) care corespunde cu batch #idx.

        Args:
            idx (int): indexul batch-ului curent

        Returns:
           tuple:  (input, target) care corespunde cu batch #idx
        """
     
               
        row = self.dataset_df.iloc[idx]
    
        
     
        img = cv2.imread(str(row['image_path']), cv2.IMREAD_GRAYSCALE)
        
        x = np.expand_dims(img, axis=0)
        print (x)
    
        j=str(row['annotations_path'])
        with open (j) as f :
            date=json.load(f)
        print (date)    
        points=date.values()
        pts = np.array(points, np.uint8)
        img=np.zeros((1024,1024,3), np.uint8)
        
        filled = cv2.fillPoly(img, pts = [pts], color =(255,255,255))
        
        print (filled)
   
        
        y = np.expand_dims(filled, axis=0)
          
            
            
        return torch.as_tensor(x.copy()).float(), torch.as_tensor(y.copy()).long()