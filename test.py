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
from monai.metrics import MSEMetric
from torchmetrics import Dice , MeanSquaredError
import skimage.color

# def (dataset_df):
    
#     for i in range()


def overlap(input , pred):
    
    width = int(512)
    height = int(512)
    dsize = (width, height)
    pred=cv2.resize(pred,dsize,interpolation = cv2.INTER_AREA)
    input=cv2.resize(input,dsize,interpolation = cv2.INTER_AREA)
    dst = cv2.addWeighted(input, 0.7, pred, 0.3, 0) 
    print(dst.shape, dst.dtype, dst.min(), dst.max())
    return dst 

def overlap_3_chanels(gt , pred , input ):
   
    
    #print(input.shape, input.dtype, input.min(), input.max())
    #print(pred.shape, pred.dtype, pred.min(), pred.max())
    width = int(512)
    height = int(512)
    dsize = (width, height)
    
    pred = pred.astype(np.float32)
    gt= gt.astype(np.float32)
    
    pred=pred*255
    input=input*255
    gt = gt *255
    
    pred=cv2.resize(pred,dsize,interpolation = cv2.INTER_AREA)
    input=cv2.resize(input,dsize,interpolation = cv2.INTER_AREA)
    gt=cv2.resize(gt,dsize,interpolation = cv2.INTER_AREA)

    pred = pred.astype(np.int32)
    gt= gt.astype(np.int32)
    
    
    tp = gt & pred
    fp = ~gt & pred
    fn =  gt & ~pred
    tn = ~gt & ~pred

    print(tp.min(), tp.max(),fp.min(),fp.max(),fn.min(),fn.max())

    img=np.zeros((512,512,3), np.float32) 
    img[:,:,1] = tp
    img[:,:,2] = fp
    img [:,:,0]= fn
 
   
    input  = skimage.color.gray2rgb(input)
    print(img.min(), img.max(),img.dtype ,img.shape)
    print (input.min(), input.max(),input.dtype ,input.shape)
    dst = cv2.addWeighted(input, 0.7, img, 0.3, 0) 
    #plt.imshow(img)
    #plt.show()
    
    
    
    return dst
    
    

    
def  dice_histogram_maker(dice_forground,dice_background, path):

    plt.hist(dice_forground , [0.0,0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1.0] )
    plt.title('Frames per Image' )
    plt.xlabel('Dice Values per interval')
    plt.ylabel("Count of frames per dice value ")
    plt.savefig(f"{path}\\Histograma_dice_forground") 
    
    plt.clf()
    
    plt.hist(dice_background , [0.0,0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1.0] )
    plt.title('Frames per Image' )
    plt.xlabel('Dice Values per interval')
    plt.ylabel("Count of frames per dice value ")
    plt.savefig(f"{path}\\Histograma_dice_background") 
    
    
def mse_histogram_maker(mse,path):
    plt.hist(mse , [0.0,0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1.0] )
    plt.title('Frames per Image' )
    plt.xlabel('RMSE Values per interval')
    plt.ylabel("Count of frames per rmse value ")
    plt.savefig(f"{path}\\Histograma_rmse") 
    
    

def test(network, test_loader,dataframe, thresh=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting testing on device {device} ...")

    metric =  MeanSquaredError()
    
    #dict={'Dice_background':[],'Dice_forground':[]}

    mse_column={'MSE':[]}
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
            
            #print (current_predict.shape, tgs.shape)
            
            for batch_idx, (frame_pred ,target_pred) in enumerate(zip(current_predict , tgs)): 
                frame_pred = frame_pred.int()    
                mse = metric(frame_pred, target_pred.squeeze())
                
                #dice_CSV['Dice'].append(test_loader.dataset.csvdata())
                mse_column['MSE'].append(mse.item())
                #dict['Dice_forground'].append(dice[1].item())
                
              
            pbar.update(ins.shape[0])
        
        mse = metric.compute()
        
        print(f'[INFO] Test MSE score is {mse*100:.2f} %')
        dataframe['MSE']=mse_column['MSE']
        #dataframe['Dice_background']=dict['Dice_background']
        #dataframe['Dice_forground']=dict['Dice_forground']
        return dataframe
   
def main():
    
    config = None
    with open('config.yaml') as f: # reads .yml/.yaml files
        config = yaml.safe_load(f)
        
    yml_data=yaml.dump(config)
    directory =f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
    parent_dir =r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index11282022_1227"
    path=pt.Path(parent_dir)/directory
    path.mkdir(exist_ok=True)

    f= open(f"{path}\\yaml_config.yml","w+")
    f.write(yml_data)    
    f.close()
    path_construct=glob.glob(r"E:\__RCA_bif_detection\data\*")
    
    #path_list=create_dataset_csv(path_construct)
    #dataset_df = pd.DataFrame(path_list)  
   
        
    #dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
    #print (dataset_df.head(3))
    
    
    dataset_df=pd.read_csv(config['data']['dataset_csv'])  
    test_df = dataset_df.loc[dataset_df["subset"] == "test", :]
    test_ds = AngioClass(test_df, img_size=config["data"]["img_size"])
    #print(test_ds[0])
    
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["train"]["bs"], shuffle=False)
    
    
    #print (AngioClass.__csvdata__())
    
    network = torch.load(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index11282022_1227\Weights\Weights_my_model11292022_0156_e400.pt")

    #print(f"# Test: {len(test_ds)}")


    test_set_CSV=test(network, test_loader,test_df, thresh=config['test']['threshold'])
   
    test_set_CSV.to_csv(f"{path}\\CSV_TEST.csv")
    
    mse_histogram_maker(test_set_CSV['MSE'],path)
    


    for batch_index,batch in enumerate(test_loader):
        x, y = iter(batch)
        
        network.eval()
        x=x.type(torch.cuda.FloatTensor)
        print (x.shape)
        y_pred = network(x.to(device='cuda:0'))
       # print( y_pred.shape)
       # print (len(x),x.shape)
       
        for  step ,(input,gt, pred) in enumerate(zip(x,y,y_pred)):
            #print (pred.shape, img.shape,gt.shape)
            #print(pred.shape)
            pred = F.softmax(pred,dim=0)[1].detach().cpu().numpy()
            # print(pred.shape, pred.min(), pred.max())
            pred[pred > config['test']['threshold']] = 1
            pred[pred <= config['test']['threshold']] = 0    
            
           
            np_input=input.cpu().detach().numpy()
            np_gt=gt.cpu().detach().numpy()
           
          
            #print (pred.shape,input.shape)
            print(np_gt.dtype,np_gt.shape, np_gt.min(),np_gt.max())
            #dst=overlap(np_input[0],pred)
            overlap_colors = overlap_3_chanels(np_gt[0] , pred , np_input[0])
            
            # plt.axis('off')
            # plt.title('Prediction ')
            # plt.imshow(pred, cmap='gray')      
            # plt.savefig(f"{path}\\Măști_frame{batch_index}_{step}.png") 
            
            # plt.title('Overlap ')
            # plt.imshow(np_input[0],cmap='gray',interpolation=None) 
            # plt.imshow(pred,cmap='jet',interpolation=None,alpha=0.8)
            # plt.savefig(f"{path}\\OVERLAP_frame{batch_index}_{step}.png") 
            
            #cv2.imwrite(os.path.join(path, 'OVERLAP'+'_'+str(batch_index)+'_'+str(step)+'.png'),dst)
            cv2.imwrite(os.path.join(path, 'OVERLAP_Colored'+'_'+str(batch_index)+'_'+str(step)+'.png'),overlap_colors)
            cv2.imwrite(os.path.join(path, 'PREDICTIE'+'_'+str(batch_index)+'_'+str(step)+'.png'),pred*255)
           
       
if __name__ == "__main__":
    main()
