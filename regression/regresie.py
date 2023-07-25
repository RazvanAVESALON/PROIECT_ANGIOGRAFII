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
import monai.transforms as TR
import torchvision.transforms.functional as TF 
from skimage.color import gray2rgb


class RegersionClass(torch.utils.data.Dataset):
    def __init__(self, dataset_df, img_size,geometrics_transforms=None,pixel_transforms=None):
        self.dataset_df = dataset_df.reset_index(drop=True)
        self.img_size = tuple(img_size)
        self.pixel_transforms=pixel_transforms
        self.geometrics_transforms=geometrics_transforms

    def __len__(self):

        return len(self.dataset_df)
    
    def csvdata (self,idx):
        
        
        patient=self.dataset_df['patient'][idx]
        acquisition=self.dataset_df['acquisition'][idx]
        frame=self.dataset_df['frames'][idx]
        header=self.dataset_df['angio_loader_header'][idx]
        annotations=self.dataset_df['annotations_path'][idx]
        
        return patient, acquisition, frame , header ,annotations
    
 
    def crop_colimator(self, frame, info):
        img = frame.astype(np.float32)
        in_min = 0
        in_max = 2 ** info['BitsStored'] - 1
        out_min = 0
        out_max = 255
        if in_max != out_max:
            img = img.astype(np.float32)
            img = (img - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
            img = np.rint(img)
            img.astype(np.uint8)
            
        # crop collimator
        img_edge = info['ImageEdges']
        img_c = img[..., img_edge[2]:img_edge[3]+1, img_edge[0]:img_edge[1]+1]
        
        return img_c 

    
    def __getitem__(self, idx):
        img = np.load(self.dataset_df['images_path'][idx])['arr_0']
        frame_param = self.dataset_df['frames'][idx]
        new_img = img[frame_param]
        
        with open(self.dataset_df['angio_loader_header'][idx]) as f:
            angio_loader = json.load(f)
            
        with open(self.dataset_df['annotations_path'][idx]) as f:
            clipping_points = json.load(f)
        
        bifurcation_point = clipping_points[str(frame_param)]
        croped_colimator_img=self.crop_colimator(new_img,angio_loader)
        
      
        bifurcation_point[1]=bifurcation_point[1]-angio_loader['ImageEdges'][0]
        bifurcation_point[0]=bifurcation_point[0]-angio_loader['ImageEdges'][2]
        # print('Bifurcation point : ',bifurcation_point)
        # print(bifurcation_point[1], bifurcation_point[0])
        # print(croped_colimator_img.shape)
        # black=np.zeros(croped_colimator_img.shape)
        # masked_gt=cv2.circle(black, (int(bifurcation_point[1]),int(bifurcation_point[0])), 5, [255,0,0], -1) 
        # plt.subplot(1,2,1)
        # plt.imshow(croped_colimator_img, cmap="gray")
        # plt.subplot(1,2,2)
        # plt.imshow(masked_gt , cmap="gray")
        # plt.show()
       
        
        # new_img = cv2.resize(croped_colimator_img, self.img_size, interpolation=cv2.INTER_AREA)
        # print (croped_colimator_img.shape)
        # bifurcation_point[0]=bifurcation_point[0]*self.img_size[0]/croped_colimator_img.shape[0]
        # bifurcation_point[1]=bifurcation_point[1]*self.img_size[1]/croped_colimator_img.shape[1]
        # bifurcation_point=np.array(bifurcation_point)
        # print(new_img.shape, type(new_img))
        
        # black=np.zeros((512,512))
        # masked_gt=cv2.circle(black, (int(bifurcation_point[1]),int(bifurcation_point[0])), 5, [255,0,0], -1) 
        # plt.subplot(1,2,1)
        # plt.imshow(new_img, cmap="gray")
        # plt.subplot(1,2,2)
        # plt.imshow(masked_gt , cmap="gray")
        # plt.show()
        new_img=croped_colimator_img
        if self.geometrics_transforms !=None:
            list_of_keypoints=[]
            list_of_keypoints.append(tuple(bifurcation_point))
            transformed=self.geometrics_transforms (image=new_img,keypoints=list_of_keypoints)
            new_img=transformed['image']
            bifurcation_point=transformed['keypoints'][0]
        
        
        
        new_img=new_img.astype(np.uint8)
        
        if self.pixel_transforms != None:

            list_of_keypoints=[]
            list_of_keypoints.append(tuple(bifurcation_point))
            transformed=self.pixel_transforms (image=new_img,keypoints=list_of_keypoints)
            new_img=transformed['image']
            bifurcation_point=transformed['keypoints'][0]
            
        
        black=np.zeros((512,512))
        masked_gt=cv2.circle(black, (int(bifurcation_point[1]),int(bifurcation_point[0])), 5, [255,0,0], -1) 
        # plt.subplot(1,2,1)
        # plt.imshow(new_img, cmap="gray")
        # plt.subplot(1,2,2)
        # plt.imshow(masked_gt , cmap="gray")
        # plt.show()
        bf=list(bifurcation_point)
        new_img=new_img*1/255 
        bf[0] = bf[0]*(1/255) 
        bf[1] = bf[1]*(1/255)
        
        print ('bf',bf) 
        img_3d=np.zeros((3,new_img.shape[0],new_img.shape[1]))
        img_3d[0,:,:]=new_img
        img_3d[1,:,:]=new_img
        img_3d[2,:,:]=new_img
        tensor_y = torch.from_numpy(np.array(bf))
        tensor_x = torch.from_numpy(img_3d)
        
      
        #print (tensor_x.min(),tensor_y.min(),tensor_x.max(),tensor_y.max())
       
        #plt.imshow(tensor_x, cmap="gray")
        #plt.show()
        #plt.imshow(tensor_y[0] , cmap="gray")
        #plt.show()

        return tensor_x.float(), tensor_y.float(), idx
    