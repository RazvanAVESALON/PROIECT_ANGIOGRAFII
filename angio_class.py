from __future__ import annotations
from hashlib import new
from statistics import geometric_mean
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
    def __init__(self, dataset_df, img_size,geometric_transforms,pixel_transforms):
        self.dataset_df = dataset_df.reset_index(drop=True)
        self.img_size = tuple(img_size)
        self.geometric_transforms=geometric_transforms
        self.pixel_transforms=pixel_transforms

    def __len__(self):

        return len(self.dataset_df)

    def __getitem__(self, idx):
        img = np.load(self.dataset_df['images_path'][idx])['arr_0']
       

        frame_param = self.dataset_df['frames'][idx]
        new_img = img[frame_param]

        new_img = cv2.resize(img[frame_param], self.img_size, interpolation=cv2.INTER_AREA)
        new_img = new_img*1/255

        with open(self.dataset_df['annotations_path'][idx]) as f:
            clipping_points = json.load(f)

        target = np.zeros(img.shape, dtype=np.uint8)
        target[frame_param] = cv2.circle(target[frame_param], [clipping_points[str(frame_param)][1], clipping_points[str(frame_param)][0]], 8, [255, 255, 255], -1)
        new_target = cv2.resize( target[frame_param], self.img_size, interpolation=cv2.INTER_AREA)
     
        tensor_x = torch.from_numpy(new_img)
        tensor_x = self.geometric_transforms(tensor_x)
        tensor_x=  self.pixel_transforms(tensor_x)
        
        tensor_y=torch.from_numpy(new_target)
        tensor_y=self.geometric_transforms(tensor_y)
        # plt.imshow(tensor_x[0], cmap="gray")
        # plt.show()
        # plt.imshow(tensor_y[0], cmap="gray")
        # plt.show()

        return tensor_x, tensor_y


def plot_acc_loss(result,path):
    acc = result['acc']['train']
    loss = result['loss']['train']
    val_acc = result['acc']['valid']
    val_loss = result['loss']['valid']
    
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
    plt.subplot(122)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.savefig(f"{path}\\Curbe de învățare")