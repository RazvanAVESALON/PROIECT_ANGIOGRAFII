
import cv2
from skimage.feature import blob_doh
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

def blob_detector(img):

    img = rgb2gray(img)
    img = img.astype(np.float64)
    # print(image_gray,image_gray.shape,image_gray.dtype,image_gray.max())
    blobs_doh = blob_doh(img, max_sigma=30, threshold=.01)
    cords_list = {"x": [], "y": []}
    #print (blobs_doh)
    fig, axes = plt.subplots()
    plt.title('Determinant of Hessian')
    plt.imshow(img)
    for blob in blobs_doh:
        #print (blob)
        y, x, r = blob
        cords_list["x"].append(x)
        cords_list["y"].append(y)
        c = plt.Circle((x, y), r, color='blue', linewidth=2, fill=False)
        axes.add_patch(c)

    #plt.tight_layout()
    plt.show()
    return cords_list


def main():
    img = cv2.imread(
        r"C:\Users\razav\OneDrive\Desktop\Poze documentatie licenta\PREDICTIE_00e91486f2fd40f198e7e09216937155_48809182-4.png")
    print(img, img.shape, img.dtype)
    list = blob_detector(img)


if __name__ == "__main__":
    main()
