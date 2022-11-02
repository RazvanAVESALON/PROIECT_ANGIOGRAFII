from asyncio.windows_events import NULL
from importlib.resources import path
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


na=0
non=0
path_construct = glob.glob(config["data"]['data_path'])

for patient in path_construct:

    x = glob.glob(os.path.join(patient, r"*"))

    for acquisiton in x:
        img = os.path.join(acquisiton, "frame_extractor_frames.npz")
        annotations=os.path.join(acquisiton,"clipping_points.json")
        with open (annotations) as f :
            clipping_points=json.load(f)
        img = np.load(img)['arr_0']
      
        na=na+img.shape[0]
        non=non+len(clipping_points)
        
        
frameuri_neadnotate=na-non

frames=[frameuri_neadnotate, non] 

print (frames)
plt.hist (frames)   
plt.title('Frameuri' )
plt.savefig(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Histograma_frameuri")  
print (frameuri_neadnotate, non) 
            
            