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
from UNetModel import UNet
from configurare_data import create_dataset_csv, split_dataset
from lungs_class import LungSegDataset, plot_acc_loss
import os
from datetime import datetime


def train(network, train_loader, valid_loader, criterion, opt, epochs=100, thresh=0.5, weights_dir='weights'):
    total_loss = {'train': [], 'valid': []}
    total_acc = {'train': [], 'valid': []}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting training on device {device} ...")

    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }
    metric = torchmetrics.Accuracy()

    network.to(device)
    criterion.to(device)

    for ep in range(epochs):

        print(f"[INFO] Epoch {ep}/{epochs - 1}")

        print("-" * 20)
        for phase in ['train', 'valid']:
            running_loss = 0.0

            if phase == 'train':
                network.train()  # Set model to training mode
            else:
                network.eval()   # Set model to evaluate mode

            with tqdm(desc=phase, unit=' batch', total=len(loaders[phase].dataset)) as pbar:
                for data in loaders[phase]:
                    ins, tgs = data
                    ins = ins.to(device)
                    tgs = tgs.to(device)
                    #print (ins.size(),tgs.size())
                    # seteaza toti gradientii la zero, deoarece PyTorch acumuleaza valorile lor dupa mai multe backward passes
                    opt.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # se face forward propagation -> se calculeaza predictia
                        output = network(ins)
                        # print(output.size())

                        #second_output = Variable(torch.argmax(output,1).float(),requires_grad=True).cuda()

                        # se calculeaza eroarea/loss-ul
                        loss = criterion(output, tgs.squeeze())

                        # deoarece reteaua nu include un strat de softmax, predictia finala trebuie calculata manual
                        current_predict = (F.softmax(output, dim=1)[
                                           :, 1] > thresh).float()

                        if 'cuda' in device.type:
                            current_predict = current_predict.cpu()
                            current_target = tgs.cpu().type(torch.int).squeeze()
                        else:
                            current_predict = current_predict
                            current_target = tgs.type(torch.int).squeeze()

                        # print(current_predict.shape, current_target.shape)
                        # print(current_predict.dtype, current_target.dtype)
                        acc = metric(current_predict, current_target)
                        # print(f"\tAcc on batch {i}: {acc}")

                        if phase == 'train':
                            # se face backpropagation -> se calculeaza gradientii
                            loss.backward()
                            # se actualizează weights-urile
                            opt.step()

                    running_loss += loss.item() * ins.size(0)
                    # print(running_loss, loss.item())

                    if phase == 'valid':
                        # salvam ponderile modelului dupa fiecare epoca
                        # torch.save(network, 'my_model.pth')
                        model_path = f"{weights_dir}\\model_epoch{ep}.pth"
                        torch.save({'epoch': ep,

                                    'loss': total_loss,
                                    }, model_path)

                    pbar.update(ins.shape[0])

                # Calculam loss-ul pt toate batch-urile dintr-o epoca
                total_loss[phase].append(
                    running_loss/len(loaders[phase].dataset))

                # Calculam acuratetea pt toate batch-urile dintr-o epoca
                acc = metric.compute()
                total_acc[phase].append(acc)

                postfix = f'error {total_loss[phase][-1]:.4f} accuracy {acc*100:.2f}%'
                pbar.set_postfix_str(postfix)

                # Resetam pt a acumula valorile dintr-o noua epoca
                metric.reset()

        return {'loss': total_loss, 'acc': total_acc}


directory = f"Experiment{datetime.now().strftime('%m%d%Y_%H%M')}"

parent_dir = os.getcwd()  # get current working directory
path = os.path.join(parent_dir, directory)
os.mkdir(path)
dir = "Weights"
path = os.path.join(path, dir)
os.mkdir(path)

print(f"pyTorch version {torch.__version__}")
print(f"torchvision version {torchvision.__version__}")
print(f"torchmetrics version {torchmetrics.__version__}")
print(f"CUDA available {torch.cuda.is_available()}")

config = None
with open('config.yaml') as f:  # reads .yml/.yaml files
    config = yaml.safe_load(f)


dataset_df = create_dataset_csv(config["data"]["images_dir"],
                                config["data"]["right_masks_dir"],
                                config["data"]["left_masks_dir"],
                                config["data"]["data_csv"])

dataset_df = split_dataset(
    dataset_df, split_per=config['data']['split_per'], seed=1)
dataset_df.head(3)

data_ds = LungSegDataset(dataset_df, img_size=config["data"]["img_size"])
x, y = data_ds[0]
print(x.shape, y.shape)

f, axs = plt.subplots(1, 2)
axs[0].axis('off')
axs[0].set_title("Input")
axs[0].imshow(x[0].numpy(), cmap="gray")

axs[1].axis('off')
axs[1].set_title("Mask")
axs[1].imshow(y[0].numpy(), cmap="gray")

network = UNet(n_channels=1, n_classes=2)
print(network)

train_df = dataset_df.loc[dataset_df["subset"] == "train", :]
train_ds = LungSegDataset(train_df, img_size=config["data"]["img_size"])
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=config['train']['bs'], shuffle=True)

valid_df = dataset_df.loc[dataset_df["subset"] == "valid", :]
valid_ds = LungSegDataset(valid_df, img_size=config["data"]["img_size"])
valid_loader = torch.utils.data.DataLoader(
    valid_ds, batch_size=config['train']['bs'], shuffle=False)

print(f"# Train: {len(train_ds)} # Valid: {len(valid_ds)}")

criterion = torch.nn.CrossEntropyLoss()

if config['train']['opt'] == 'Adam':
    opt = torch.optim.Adam(network.parameters(), lr=config['train']['lr'])
elif config['train']['opt'] == 'SGD':
    opt = torch.optim.SGD(network.parameters(), lr=config['train']['lr'])

history = train(network, train_loader, valid_loader, criterion, opt,
                epochs=config['train']['epochs'], thresh=config['test']['threshold'], weights_dir=path)

plot_acc_loss(history)
