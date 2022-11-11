from pickle import NONE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import random
import yaml
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchmetrics 

from tqdm import tqdm
from UNet import UNet
from torchmetrics.functional import dice_score
from configurare_data import create_dataset_csv , split_dataset
from test_function import test
from datetime import datetime
import os 
import cv2
import glob
from angio_class import AngioClass

from torchmetrics import Dice

# def (dataset_df):
    
#     for i in range()


def test(network, test_loader, thresh=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting testing on device {device} ...")

    metric =  Dice(average='micro')
    dice_CSV={'Dice':[]}
    network.eval()
    with tqdm(desc='test', unit=' batch', total=len(test_loader.dataset)) as pbar:
        
        for data in test_loader:
            ins, tgs= data
            network.to(device)
            ins = ins.to(device)
            tgs = tgs.to('cpu')
            output = network(ins)
         
            current_predict = (F.softmax(output, dim=1)[:, 1] > thresh)

           
            if 'cuda' in device.type:
                current_predict = current_predict.cpu()
            
            print (current_predict.shape, tgs.shape)
            for (frame_pred ,target_pred) in zip(current_predict , tgs):  
                dice = metric(frame_pred, target_pred.squeeze())
                dice_CSV['Dice'].append(dice)
            pbar.update(ins.shape[0])
        
        dice = metric.compute()
        
        print(f'[INFO] Test accuracy is {dice*100:.2f} %')
        return dice_CSV


# def csvdata (dataset_df):
#     data={'patients':[],'acquisition':[],'frames':[]}
#     for id in range(len(dataset_df['patient'])):
#         data['patients'].append(dataset_df['patient'][id])
#         data['acquisition'].append(dataset_df['acquisition'][id])
#         data['frame'].append(dataset_df['frames'][id])

    
#     return data

   
def main():
    
    config = None
    with open('config.yaml') as f: # reads .yml/.yaml files
        config = yaml.safe_load(f)
        
    yml_data=yaml.dump(config)
    directory =f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
    parent_dir =r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index11072022_2316"
    path=pt.Path(parent_dir)/directory
    path.mkdir(exist_ok=True)

    f= open(f"{path}\\yaml_config.yml","w+")
    f.write(yml_data)    

    path_construct=glob.glob(r"E:\__RCA_bif_detection\data\*")
    
    #path_list=create_dataset_csv(path_construct)
    #dataset_df = pd.DataFrame(path_list)  
   
        
    #dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
    #print (dataset_df.head(3))
    
    
    dataset_df=pd.read_csv(config['data']['dataset_csv'])  
    test_df = dataset_df.loc[dataset_df["subset"] == "test", :]
    test_ds = AngioClass(test_df, img_size=config["data"]["img_size"])
    print(test_ds[0])
    
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["train"]["bs"], shuffle=False)
    
    print (test_loader.dataset[0])
    #print (AngioClass.__csvdata__())
    
    network = torch.load(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index11072022_2316\Weights\my_model11082022_0223_e90.pt")

    print(f"# Test: {len(test_ds)}")


    test_set_CSV=test(network, test_loader, thresh=config['test']['threshold'])
    
    test_df['Dice']=test_set_CSV['Dice']
    
    test_CSV=pd.DataFrame(test_df)
    test_CSV.to_csv(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index11072022_2316\CSV_TEST.csv")

   
    for batch_index,batch in enumerate(test_loader):
        x, y = iter(batch)
        
        network.eval()
        x=x.type(torch.cuda.FloatTensor)
        y_pred = network(x.to(device='cuda:0'))
        print( y_pred.shape)

        
        fig, axs = plt.subplots(3, figsize=(50, 50))
       
        print (len(x),x.shape)
       
        for  step , (img, gt, pred) in enumerate(zip(x, y, y_pred)):
            print (pred.shape, img.shape,gt.shape)
            axs[0].axis('off')
            axs[0].set_title('Input')
            axs[0].imshow(img[0].cpu(), cmap='gray')
    
            axs[1].axis('off')
            axs[1].set_title('Ground truth')
            axs[1].imshow(gt[0], cmap='gray')

            # print(pred.shape)
            pred = F.softmax(pred,dim=0)[1].detach().cpu().numpy()
            # print(pred.shape, pred.min(), pred.max())
            pred[pred > config['test']['threshold']] = 1
            pred[pred <= config['test']['threshold']] = 0    
            pred = pred.astype(np.uint8)
            
            print (pred.shape)
                    
            axs[2].axis('off')
            axs[2].set_title('Prediction ')
            print (pred.shape)
            axs[2].imshow(pred, cmap='gray')
                    
            plt.savefig(f"{path}\\Măști_frame{batch_index}_{step}.png") 
                
        
            
        


       
if __name__ == "__main__":
    main()
