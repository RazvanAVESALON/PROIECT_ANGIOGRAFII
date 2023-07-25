
import cv2
from skimage.feature import blob_doh
import matplotlib.pyplot as plt
import numpy as np


def blob_detector(img):

    #schimbarea tipului de date a imagini 
    img = img.astype(np.float64)
    #inițializarea funcției de detecție a puncte
    blobs_doh = blob_doh(img, max_sigma=30, threshold=.01)
    cords_list = {"x": [], "y": []}
 
    for blob in blobs_doh:
      
        y, x, r = blob
        # crearea listă cu punctele de bifucație
        cords_list["x"].append(x)
        cords_list["y"].append(y)
       
  
    return cords_list


def main():
    img = cv2.imread(
        r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\PREDICTIE_14_3.png")
    print(img, img.shape, img.dtype)
    list = blob_detector(img)
    


if __name__ == "__main__":
    main()
