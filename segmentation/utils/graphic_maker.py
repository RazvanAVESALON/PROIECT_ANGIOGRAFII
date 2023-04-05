import json
import yaml
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
config = None
with open('config.yaml') as f:  # reads .yml/.yaml files
    config = yaml.safe_load(f)


def calc_frame_uri(x):
    for y in x:
        suma = suma+y

    return suma


na = 0
non = 0
path_construct = glob.glob(config["data"]['data_path'])

frames = []
for patient in path_construct:

    x = glob.glob(os.path.join(patient, r"*"))

    for acquisiton in x:
        img = os.path.join(acquisiton, "frame_extractor_frames.npz")
        annotations = os.path.join(acquisiton, "clipping_points.json")
        with open(annotations) as f:
            clipping_points = json.load(f)
        img = np.load(img)['arr_0']
        x = img.shape[0]
        frames.append(x)


df = pd.read_csv(
    r'D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\CSV_angiografii.csv')

dataset = ["Train", "Validation ", "Test"]
test = df.loc[df["subset"] == "test", :]
train = df.loc[df["subset"] == "train", :]
valid = df.loc[df["subset"] == "valid", :]

patient__test = test['patient'].unique()
patient__train = train['patient'].unique()
patient__valid = valid['patient'].unique()

frame_test = test['frames']
frame_train = train['frames']
frame_valid = valid['frames']

total_frames_test = sum(frame_test)
total_frames_train = sum(frame_train)
total_frames_valid = sum(frame_valid)


height = [len(patient__train), len(patient__valid), len(patient__test)]
height_frames = [total_frames_train, total_frames_valid, total_frames_test]

# plt.bar(dataset,height)
# plt.title("PATIENTS PER SPLITS ")
# plt.xlabel("SPLITS")
# plt.ylabel("PATIENTS")
# plt.savefig(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Barplot_patietspersplits")


# plt.bar(dataset,height_frames)
# plt.title("FRAMES PER SPLITS ")
# plt.xlabel("SPLITS")
# plt.ylabel("FRAMES")
# plt.savefig(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Barplot_FRAMESperSPLITS")


print(frames)
plt.hist(frames, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
plt.title('Frames per Image')
plt.xlabel('Count 0f frames per Image')
plt.ylabel(" NUMBER OF FRAMES Frames")
plt.savefig(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Histograma_frameuri")
