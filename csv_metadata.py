import glob
import os 
import pandas as pd 
import json

path_construct=glob.glob(r'E:\__RCA_bif_detection\data\*')
path_construct2=glob.glob(r'E:\__RCA_bif_detection\pacienti_noi\*')
path_construct3=glob.glob(r'E:\__RCA_bif_detection\pacienti_11jan\*')
path_list = {"patient": [], "acquisition": [],"MagFactor":[], "Img_size":[], 'Img_spacing':[]}
# frame_list={"frames"}

for patient in path_construct:
        #print (image)
        # x=os.path.join(image,r"*")

    x = glob.glob(os.path.join(patient, r"*"))
  
    
    for acquisiton in x:
        img = os.path.join(acquisiton, "frame_extractor_frames.npz")
        annotations = os.path.join(acquisiton, "clipping_points.json")
        
       
        angio_leader = os.path.join(acquisiton, "angio_loader_header.json")
        path_list["acquisition"].append(acquisiton)
        path_list["patient"].append(patient)
        print (angio_leader)
        with open(angio_leader) as f:
            angio_loader = json.load(f)
        path_list["MagFactor"].append(angio_loader['MagnificationFactor'])
        path_list["Img_spacing"].append(angio_loader['ImageSpacing'])
        path_list["Img_size"].append(angio_loader['ImageSize'])
        
        
for patient in path_construct2:
        #print (image)
        # x=os.path.join(image,r"*")

    x = glob.glob(os.path.join(patient, r"*"))
        # print (x)
    for acquisiton in x:
        img = os.path.join(acquisiton, "frame_extractor_frames.npz")
        annotations = os.path.join(acquisiton, "clipping_points.json")
        angio_leader = os.path.join(acquisiton, "angio_loader_header.json")
        
        path_list["acquisition"].append(acquisiton)
        path_list["patient"].append(patient)
        
        with open(angio_leader) as f:
            angio_loader = json.load(f)
        path_list["MagFactor"].append(angio_loader['MagnificationFactor'])
        path_list["Img_spacing"].append(angio_loader['ImageSpacing'])
        path_list["Img_size"].append(angio_loader['ImageSize'])

        
for patient in path_construct3:
        #print (image)
        # x=os.path.join(image,r"*")

    x = glob.glob(os.path.join(patient, r"*"))
        # print (x)
    for acquisiton in x:
        img = os.path.join(acquisiton, "frame_extractor_frames.npz")
        annotations = os.path.join(acquisiton, "clipping_points.json")
        angio_leader = os.path.join(acquisiton, "angio_loader_header.json")
        path_list["acquisition"].append(acquisiton)
        path_list["patient"].append(patient)
        
        with open(angio_leader) as f:
            angio_loader = json.load(f)
        path_list["MagFactor"].append(angio_loader['MagnificationFactor'])
        path_list["Img_spacing"].append(angio_loader['ImageSpacing'])
        path_list["Img_size"].append(angio_loader['ImageSize'])
df=pd.DataFrame(path_list)
df.to_csv(r'D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\csv_metadata.csv')

                
            
