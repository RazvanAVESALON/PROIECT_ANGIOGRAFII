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
    def __init__(self, image_path_list):
        self.image_path_list = image_path_list
      
    
    def __len__(self):
      
        return len(self.image_path_list) 

    def __getitem__(self, idx):
        """Returneaza un tuple (input, target) care corespunde cu batch #idx.

        Args:
            idx (int): indexul batch-ului curent

        Returns:
           tuple:  (input, target) care corespunde cu batch #idx
        """
        list_of_img:[]
        list_of_annotaion:[] 
        for i in self.image_path_list: 
            
            img = cv2.imread(os.path.join(i,f"frame_extractor_frames.npz"), cv2.IMREAD_GRAYSCALE)
            list_of_img.append(img)
            with open(os.path.join(i, "clipping_points.json"), "r") as f:
                clipping_points = json.load(f)
            
            list_of_annotaion.append(clipping_points)
        
        list_of_image_tensors:[]
        for image in list_of_image_tensors:
            x=torch.from_numpy(image)
            list_of_image_tensors.append(x)
        
        list_of_annotation_tensors:[]
        for image in list_of_annotaion:
            x=torch.from_numpy(image)
            list_of_annotation_tensors.append(x)
            
            
        return list_of_image_tensors,list_of_annotation_tensors