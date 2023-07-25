import json
import yaml
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt


def calc_frame_uri(x):
    for y in x:
        suma = suma+y

    return suma


na = 0
non = 0
path_construct = glob.glob(r"E:\\__RCA_bif_detection\\data\\*")

frames = []
acq= []
for patient in path_construct:

    x = glob.glob(os.path.join(patient, r"*"))
    nr=0
    for acquisiton in x:
        nr+=1
        img = os.path.join(acquisiton, "frame_extractor_frames.npz")
        annotations = os.path.join(acquisiton, "clipping_points.json")
        with open(annotations) as f:
            clipping_points = json.load(f)
        img = np.load(img)['arr_0']
        x = img.shape[0]
        frames.append(x)
    acq.append(nr)



df = pd.read_csv(
    r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\CSV_angiografii_date_adaugate.csv")

dataset = ["Train", "Validation ", "Test"]
test = df.loc[df["subset"] == "test", :]
train = df.loc[df["subset"] == "train", :]
valid = df.loc[df["subset"] == "valid", :]

patient__test = test['patient'].unique()
patient__train = train['patient'].unique()
patient__valid = valid['patient'].unique()
print (len(patient__test),len(patient__train),len(patient__valid)) 
frame_test = test['frames']
frame_train = train['frames']
frame_valid = valid['frames']

total_frames_test = len(frame_test)
total_frames_train = len(frame_train)
total_frames_valid = len(frame_valid)


height = [len(patient__train), len(patient__valid), len(patient__test)]
height_frames = [total_frames_train, total_frames_valid, total_frames_test]

plt.clf()
plt.bar(dataset,height)
plt.title("Împărțirea Pacienților")
plt.xlabel("Seturi")
plt.ylabel("Pacienți")
plt.savefig(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\plots\Barplot_Pacienti_per_split")

plt.clf()
plt.bar(dataset,height_frames)
plt.title("Împărțirea cadrelor")
plt.xlabel("Seturi")
plt.ylabel("Număr de cadre")
plt.savefig(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\plots\Barplot_frames_per_SPLITS")

plt.clf()

print(frames)
plt.hist(frames, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
plt.title('Cadre pe Angiografie')
plt.xlabel('Număr de cadre pe angiografie ')
plt.ylabel(" Număr de angiografii")
plt.savefig(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\plots\Histograma_frameuri_pe_angiografie")

print(acq)
plt.clf()
plt.hist(acq, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15])
plt.title('Achiziții pe pacient')
plt.xlabel('Număr de achiziții pe pacient ')
plt.ylabel(" Număr de pacienți")
plt.savefig(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\plots\Histograma_acq_pe_angiografie")
