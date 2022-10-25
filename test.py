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

from configurare_data import create_dataset_csv , split_dataset
from test_function import test
from datetime import datetime
import os 
import cv2
import glob
from angio_class import AngioClass


def test(network, test_loader, thresh=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting testing on device {device} ...")

    metric = torchmetrics.Accuracy()

    network.eval()
    with tqdm(desc='test', unit=' batch', total=len(test_loader.dataset)) as pbar:
        for data in test_loader:
            ins, tgs = data
            ins = ins.to(device)
            tgs = tgs.to('cpu')

            output = network(ins)
            current_predict = (F.softmax(output, dim=1)[:, 1] > thresh)

            if 'cuda' in device.type:
                current_predict = current_predict.cpu()
                
            acc = metric(current_predict, tgs.squeeze())
            pbar.update(ins.shape[0])
        
        acc = metric.compute()
        print(f'[INFO] Test accuracy is {acc*100:.2f} %')



   
def main():
    
    config = None
    with open('config.yaml') as f: # reads .yml/.yaml files
        config = yaml.safe_load(f)
        
    yml_data=yaml.dump(config)
    directory =f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
    parent_dir =r'D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index10122022_1556'
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)

    f= open(f"{path}\\yaml_config.txt","w+")
    f.write(yml_data)    

    path_construct=glob.glob(r"E:\__RCA_bif_detection\data\*")
    path_list=create_dataset_csv(path_construct)
    dataset_df = pd.DataFrame(path_list)  
    dataset_df.to_csv(r'D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\CSV_angiografii.csv')  
        
    dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
    print (dataset_df.head(3))
    dataset_df.to_csv(r'D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\CSV_angiografii.csv')  

    test_df = dataset_df.loc[dataset_df["subset"] == "test", :]
    test_ds = AngioClass(test_df, img_size=config["data"]["img_size"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["train"]["bs"], shuffle=False)

    network = torch.load(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\Experiment_Dice_index10122022_1556\Weights\my_model10122022_1647_e5.pt")

    print(f"# Test: {len(test_ds)}")


    test(network, test_loader, thresh=config['test']['threshold'])

    x, y = next(iter(test_loader))
    network.eval()
    y_pred = network(x.to(device='cuda'))
    y_pred.shape

    nr_exs = 4 # nr de exemple de afisat
    fig, axs = plt.subplots(nr_exs, 3, figsize=(10, 10))
    for i, (img, gt, pred) in enumerate(zip(x[:nr_exs], y[:nr_exs], y_pred[:nr_exs])):
        axs[i][0].axis('off')
        axs[i][0].set_title('Input')
        axs[i][0].imshow(img[0], cmap='gray')

        axs[i][1].axis('off')
        axs[i][1].set_title('Ground truth')
        axs[i][1].imshow(gt[0], cmap='gray')

        # print(pred.shape)
        pred = F.softmax(pred, dim=0)[1].detach().cpu().numpy()
        # print(pred.shape, pred.min(), pred.max())
        pred[pred > config['test']['threshold']] = 1
        pred[pred <= config['test']['threshold']] = 0
        pred = pred.astype(np.uint8)
        
    
        axs[i][2].axis('off')
        axs[i][2].set_title('Prediction ')
        axs[i][2].imshow(pred, cmap='gray')
        
        
    plt.savefig(f"{path}\\Măști")     


       
if __name__ == "__main__":
    main()
