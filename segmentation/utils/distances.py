from __future__ import annotations
import numpy as np
import math
from .blob_detector import blob_detector
import cv2


def pixels2mm(coordonate_pixel_list, magnification_factor, image_spacing):

    coords_mm = {'x': [], 'y': []}
    #citire listă de coordonate 
    for x, y in zip(coordonate_pixel_list['x'], coordonate_pixel_list['y']):
        
        # aplicarea formulei de transformare pixeli -> milimetri
        img_space = np.array([image_spacing[0], image_spacing[1]])
        mm = np.array([x, y]) * img_space / magnification_factor

        # creare unei liste noi 
        coords_mm['x'].append(mm[0])
        coords_mm['y'].append(mm[1])

    return coords_mm


def mm2pixels(coordonate_mm_list, magnification_factor, image_spacing):

    coords_pixel = {'x': [], 'y': []}
    for (x, y) in zip(coordonate_mm_list['x'], coordonate_mm_list['y']):
        #print (x,y)
        pixeli = np.array(x, y)/np.array(image_spacing) * magnification_factor
        coords_pixel['x'].append(pixeli[0])
        coords_pixel['y'].append(pixeli[1])

    return coords_pixel


def calcuate_distance(gt_coordonates, pred_coordonates):
    distance = []
    # verificare dacă există predicție
    if len(gt_coordonates['x']) <= 1:
    
        # citire listă 
        for x, y in zip(pred_coordonates['x'], pred_coordonates['y']):

            # aplicarea formulei de distanță 
            print(gt_coordonates['x'], gt_coordonates['y'])
            dx = gt_coordonates['x']-x
            dy = gt_coordonates['y']-y

            print(dx, dy)
            d = math.sqrt(dx*dx+dy*dy)
            distance.append(d)

        print(distance)
    else:
        distance.append("Can't calculate Distance For this frame ( No prediction )")
        

    return distance


def main():
    pred = cv2.imread(
        r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\PREDICTIE_14_3.png")
    gt = cv2.imread(
        r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index11282022_1227\Test12112022_1345\PREDICTIE_4_0.png")

    cord_list_pred = blob_detector(pred)
    image_gray = rgb2gray(gt)
    cord_list_gt = blob_detector(image_gray)

    print(cord_list_pred)
    print(cord_list_gt)
    cord_list_gt = pixels2mm(cord_list_gt, 1.3, [0.278875, 0.278875])
    coord_list_pred = pixels2mm(cord_list_pred, 1.3, [0.278875, 0.278875])
    print(cord_list_gt)
    print(coord_list_pred)
    distance = calcuate_distance(cord_list_gt, coord_list_pred)
    print(distance)


if __name__ == "__main__":
    main()
