
import torch
import monai.transforms as TR
import numpy as np
import cv2
import os
import albumentations as A
pixel_t =  A.GaussianBlur(blur_limit=[3, 7] , sigma_limit=[0,10], always_apply=False, p=1)
#A.RandomGamma(gamma_limit= [60, 70],eps=None, always_apply=False, p=1)
#A.CLAHE(clip_limit=8, tile_grid_size=[8, 8] , always_apply=False, p=1)
        #A.GaussianBlur(blur_limit=config['train']['blur_limit'], sigma_limit=config['train']['sigma_limit'], always_apply=False, p=config['train']['p_gauss_blur']),
        #A.RandomGamma(gamma_limit=config['train']['gamma_limit'],eps=None, always_apply=False, p=config['train']['p'])
#TR.RandAdjustContrastd(keys="img", prob=1, gamma=[1.5,2])
        #TR.RandGibbsNoised(keys="img", prob=1, alpha=[0.8,0.9])
        #TR.GaussianSmoothd(keys="img", sigma=[2,3] )
        #TR.RandGibbsNoised(keys="img", prob=1, alpha=[0.6,0.8]),
       # TR.RandAdjustContrastd(keys="img", prob=1, gamma=[1.5,2]),



img = np.load(r"E:\__RCA_bif_detection\data\0a0ed26fc8ea4e298130a4512cf2b8ec\35089597\frame_extractor_frames.npz")['arr_0']
# img = img[3].astype(np.float32)
# x = np.expand_dims(img, axis=0)
# tensor_y = torch.from_numpy(x)
# data_pixel = {"img": tensor_y}
# tensor_y = pixel_t(data_pixel)['img']

# tensor_y=tensor_y.cpu().detach().numpy()

# print (img.shape, tensor_y.shape, img.max(),tensor_y.max(),type(img), type(tensor_y))

list_of_keypoints = []
list_of_keypoints.append(tuple([0,1]))
transformed = pixel_t(image=img[0], keypoints=list_of_keypoints)
new_img = transformed['image']
bifurcation_point = transformed['keypoints'][0]
new_img=new_img.astype(float)
img=img.astype(float)
h_img = cv2.hconcat([img[0],new_img])

img=cv2.resize(h_img, [1024,1024])

cv2.imwrite(os.path.join(r'D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII', "Data_aug_blur.png"),h_img)