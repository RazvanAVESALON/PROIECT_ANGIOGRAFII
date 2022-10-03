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
     
               
       # row = self.dataset_df.iloc[[idx]]
       # print (row)
        
       
        img = np.load(self.dataset_df['image_path'][idx])['arr_0']
        
        #print (img.shape)
        
        #print('IMAGINE:',img)
        #x = np.expand_dims(img, axis=0)
        
        #print ('imagine:',x)
        #print(self.dataset_df['annotations_path'][idx])
        
        with open (self.dataset_df['annotations_path'][idx]) as f :
            clipping_points=json.load(f)
        #print (clipping_points)    
        
        target=np.zeros(img.shape)
        for frame in clipping_points:
            frame_int= int(frame)
            target[frame_int]=cv2.circle(target[frame_int],[clipping_points[frame][1],clipping_points[frame][0]],8,[255,255,255],-1)
            
            
            
            #plt.imshow(img[frame_int], cmap="gray")
            #plt.imshow(target[frame_int,:,:], cmap="gray")
            #plt.imshow(vesselness[frame_int], cmap="jet", alpha=0.5)
            #plt.scatter(clipping_points[frame][1], clipping_points[frame][0], marker="x", color="white")

            #plt.show()
            #x=img[frame_int,:,:]
            #y=target[frame_int,:,:]
            #x = np.expand_dims(x, axis=0)
            #y = np.expand_dims(y, axis=0)
            
            
        return torch.as_tensor(img.copy()).float(), torch.as_tensor(target.copy()).long()

        
        
        
        
        
            
        