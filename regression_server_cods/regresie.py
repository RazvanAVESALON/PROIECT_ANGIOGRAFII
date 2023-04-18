from __future__ import annotations
import numpy as np
import json
import torch


class RegersionClass(torch.utils.data.Dataset):
    def __init__(self, dataset_df, img_size, geometrics_transforms=None, pixel_transforms=None):
        self.dataset_df = dataset_df.reset_index(drop=True)
        self.img_size = tuple(img_size)
        self.pixel_transforms = pixel_transforms
        self.geometrics_transforms = geometrics_transforms

    def __len__(self):

        return len(self.dataset_df)

    def csvdata(self, idx):

        patient = self.dataset_df['patient'][idx]
        acquisition = self.dataset_df['acquisition'][idx]
        frame = self.dataset_df['frames'][idx]
        header = self.dataset_df['angio_loader_header'][idx]
        annotations = self.dataset_df['annotations_path'][idx]

        return patient, acquisition, frame, header, annotations

    def crop_colimator(self, frame, info):
        img = frame.astype(np.float32)
        in_min = 0
        in_max = 2 ** info['BitsStored'] - 1
        out_min = 0
        out_max = 255
        if in_max != out_max:
            img = img.astype(np.float32)
            img = (img - in_min) * ((out_max - out_min) /
                                    (in_max - in_min)) + out_min
            img = np.rint(img)
            img.astype(np.uint8)

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

        croped_colimator_img = self.crop_colimator(new_img, angio_loader)

        bifurcation_point[1] = bifurcation_point[1] - \
            angio_loader['ImageEdges'][0]
        bifurcation_point[0] = bifurcation_point[0] - \
            angio_loader['ImageEdges'][2]

        new_img = croped_colimator_img
        if self.geometrics_transforms != None:
            list_of_keypoints = []
            list_of_keypoints.append(tuple(bifurcation_point))
            transformed = self.geometrics_transforms(
                image=new_img, keypoints=list_of_keypoints)
            new_img = transformed['image']
            bifurcation_point = transformed['keypoints'][0]
        new_img = new_img.astype(np.uint8)
        if self.pixel_transforms != None:
            transformed = self.pixel_transforms(image=new_img)
            new_img = transformed['image']

        bf = list(bifurcation_point)
        new_img = new_img*1/255
        bf[0] = bf[0]*(1/new_img.shape[0])
        bf[1] = bf[1]*(1/new_img.shape[1])
        img_3d = np.zeros((3, new_img.shape[0], new_img.shape[1]))
        img_3d[0, :, :] = new_img
        img_3d[1, :, :] = new_img
        img_3d[2, :, :] = new_img
        tensor_y = torch.from_numpy(np.array(bf))
        tensor_x = torch.from_numpy(img_3d)

        return tensor_x.float(), tensor_y.float(), idx
