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

frames=[]
for patient in path_construct:

    x = glob.glob(os.path.join(patient, r"*"))

    for acquisiton in x:
        img = os.path.join(acquisiton, "frame_extractor_frames.npz")
        annotations=os.path.join(acquisiton,"clipping_points.json")
        with open (annotations) as f :
            clipping_points=json.load(f)
        img = np.load(img)['arr_0']
        x=img.shape[0]
        frames.append(x)
      
       
        
        


print (frames)
plt.hist (frames,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])   
plt.title('Frameuri' )
plt.savefig(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Histograma_frameuri")  

            
            