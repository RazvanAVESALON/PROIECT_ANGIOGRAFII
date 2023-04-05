import os

import json

import numpy as  np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

def main():
    root = r"F:\RCA_bif_data\00cca518a10d41adb9476aefc38a0b69\40117765"

    angio = np.load(os.path.join(root, "frame_extractor_frames.npz"))["arr_0"]
    vesselness = np.load(os.path.join(root, "vesselness_heatmaps.npz"))["arr_0"]

    with open(os.path.join(root, "clipping_points.json"), "r") as f:
        clipping_points = json.load(f)

    for frame in clipping_points:
        frame_int = int(frame)

        # TODO: use info from angio_loader_header.json to resample to 0.27 mm

        target = np.zeros_like(vesselness[frame_int])
        target[clipping_points[frame][0], clipping_points[frame][1]] = 1
        
        sigma = 10 # this should be adjusted after resampling
        target = gaussian_filter(target, sigma) * 2 * np.pi * sigma**2

        plt.imshow(angio[frame_int], cmap="gray")
        plt.imshow(target, cmap="jet", alpha=0.5)
        plt.imshow(vesselness[frame_int], cmap="jet", alpha=0.5)
        plt.scatter(clipping_points[frame][1], clipping_points[frame][0], marker="x", color="white")

        plt.show()


    x = 1

if __name__ == "__main__":
    main()