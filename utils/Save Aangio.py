import numpy as np
import cv2
import os
import json
# img = np.load(r"E:\__RCA_bif_detection\data\0a0ed26fc8ea4e298130a4512cf2b8ec\35089597\frame_extractor_frames.npz")['arr_0']

# with open(r"E:\__RCA_bif_detection\data\0a0ed26fc8ea4e298130a4512cf2b8ec\35089597\angio_loader_header.json") as f:
#     info = json.load(f)

# img = img[0].astype(np.float32)
# in_min = 0
# in_max = 2 ** info['BitsStored'] - 1
# out_min = 0
# out_max = 255
# img_c=0
# print (in_max,out_max)
# if in_max == out_max:
#     img = img.astype(np.float32)
#     img = (img - in_min) * ((out_max - out_min) /(in_max - in_min)) + out_min
#     img = np.rint(img)
#     img.astype(np.uint8)

#     # crop collimator
#     img_edge = info['ImageEdges']
#     img_c = img[..., img_edge[2]:img_edge[3]+1, img_edge[0]:img_edge[1]+1]
#     print (img_c)

# img_c=cv2.resize(img_c ,[512,512], interpolation=cv2.INTER_AREA)

img = cv2.imread(
    r"C:\Users\razav\OneDrive\Desktop\OVERLAP_Colored_0a0ed26fc8ea4e298130a4512cf2b8ec_35083539-6.png")
img2 = cv2.imread(
    r"C:\Users\razav\OneDrive\Desktop\OVERLAP_Colored_0a0ed26fc8ea4e298130a4512cf2b8ec_35083337-5.png")



foo_img_reg = cv2.putText(
    img, 'MSE: 0.00', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
foo_img_reg = cv2.putText(
    foo_img_reg, 'Distance (mm): 0.0', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))


foo_img_seg = cv2.putText(
    img2, 'Dice: 0.93', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
foo_img_seg = cv2.putText(
    foo_img_seg, 'Distance (mm): 0.0', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
h_img = cv2.hconcat([foo_img_seg, foo_img_reg])
img = cv2.resize(h_img, [1024, 1024])

cv2.imwrite(os.path.join(
    r'D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII', "Predicții_perfecte.png"), h_img)
# with open(r"E:\__RCA_bif_detection\data\0a0ed26fc8ea4e298130a4512cf2b8ec\35089597\clipping_points.json") as f:
#     clipping_points = json.load(f)

# target = np.zeros(img_c.shape, dtype=np.uint8)
# target = cv2.circle(target, [clipping_points[str(
#             0)][1], clipping_points[str(0)][0]], 8, [255, 255, 255], -1)
# cv2.imwrite(os.path.join(r'D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII', "Mască.png"),target)
