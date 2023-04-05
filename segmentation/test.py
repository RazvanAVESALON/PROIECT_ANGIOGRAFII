
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import yaml
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import os , sys
import cv2
import glob
from angio_class import AngioClass
from torchmetrics import Dice
import skimage.color
from utils.blob_detector import blob_detector
from utils.distances import pixels2mm, calculate_distance
import json
import imageio
from utils.Distances_counter import histogram_distance

def overlap(input, pred):

    width = int(512)
    height = int(512)
    dsize = (width, height)
    pred = cv2.resize(pred, dsize, interpolation=cv2.INTER_AREA)
    input = cv2.resize(input, dsize, interpolation=cv2.INTER_AREA)
    dst = cv2.addWeighted(input, 0.7, pred, 0.3, 0)
    print(dst.shape, dst.dtype, dst.min(), dst.max())
    return dst


def overlap_3_chanels(gt, pred, input):

    #print(input.shape, input.dtype, input.min(), input.max())
    #print(pred.shape, pred.dtype, pred.min(), pred.max())
    width = int(512)
    height = int(512)
    dsize = (width, height)

    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    pred = pred*255
    input = input*255
    gt = gt * 255

    pred = cv2.resize(pred, dsize, interpolation=cv2.INTER_AREA)
    input = cv2.resize(input, dsize, interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt, dsize, interpolation=cv2.INTER_AREA)

    pred = pred.astype(np.int32)
    gt = gt.astype(np.int32)

    tp = gt & pred
    fp = ~gt & pred
    fn = gt & ~pred
    tn = ~gt & ~pred

    print(tp.min(), tp.max(), fp.min(), fp.max(), fn.min(), fn.max())

    img = np.zeros((512, 512, 3), np.float32)
    img[:, :, 1] = tp
    img[:, :, 2] = fp
    img[:, :, 0] = fn

    input = skimage.color.gray2rgb(input)
    print(img.min(), img.max(), img.dtype, img.shape)
    print(input.min(), input.max(), input.dtype, input.shape)
    dst = cv2.addWeighted(input, 0.7, img, 0.3, 0)
    # plt.imshow(img)
    # plt.show()

    return dst


def dice_histogram_maker(dice_forground, dice_background, path):

    plt.clf()

    plt.hist(dice_forground, [0.0, 0.1, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Frames per Image')
    plt.xlabel('Dice Values per interval')
    plt.ylabel("Count of frames per dice value ")
    plt.savefig(f"{path}\\Histograma_dice_forground")

    plt.clf()

    plt.hist(dice_background, [0.0, 0.1, 0.2, 0.3,
             0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Frames per Image')
    plt.xlabel('Dice Values per interval')
    plt.ylabel("Count of frames per dice value ")
    plt.savefig(f"{path}\\Histograma_dice_background")


def mse_histogram_maker(mse, path):
    plt.hist(mse, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Frames per Image')
    plt.xlabel('RMSE Values per interval')
    plt.ylabel("Count of frames per rmse value ")
    plt.savefig(f"{path}\\Histograma_rmse")


def test(network, test_loader, dataframe, thresh=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting testing on device {device} ...")

    metric = Dice(average='none', num_classes=2)

    dict = {'Dice_background': [], 'Dice_forground': [], 'Distance': [], 'Ann': [], 'ImageSpacing': [
    ], 'MagnificationFactor': [], 'Header': [], 'Patient': [], 'Acquistion': [], 'Frame': []}

    # mse_column={'MSE':[]}
    network.eval()
    with tqdm(desc='test', unit=' batch', total=len(test_loader.dataset)) as pbar:

        #print (test_loader.dataset[100])
        for data in test_loader:

            ins, tgs, index = data
            print(index)

            network.to(device)
            ins = ins.to(device)
            tgs = tgs.to('cpu')
            output = network(ins)

            current_predict = (F.softmax(output, dim=1)[:, 1] > thresh)

            if 'cuda' in device.type:
                current_predict = current_predict.cpu()

            #print (current_predict.shape, tgs.shape)

            for batch_idx, (frame_pred, target_pred) in enumerate(zip(current_predict, tgs)):

                frame_pred = frame_pred.int()
                dice_score = metric(frame_pred, target_pred.squeeze())
                #print (dice)
                #print (frame_pred.index)

                patient, acquisition, frame, header, annotations = test_loader.dataset.csvdata(
                    (index[batch_idx].numpy()))
                # print(header)

                with open(annotations) as g:
                    ann = json.load(g)

                with open(header) as f:
                    angio_loader = json.load(f)

                #print (target_pred.max(),target_pred.shape,target_pred.dtype)
                pred_cord = blob_detector(frame_pred.numpy())
                gt_cord = blob_detector(target_pred[0].numpy())

                # print("Index",index[step])
                # print(patient,frame)
                #print (ann[f'{frame}'])
                #gt_coords={'x':[ann[str(frame)][0]], 'y':[ann[str(frame)][1]]}
               # print(gt_coords, pred_cord)
                gt_coords_mm = pixels2mm(
                    gt_cord, angio_loader['MagnificationFactor'], angio_loader['ImageSpacing'])
                pred_cord_mm = pixels2mm(
                    pred_cord, angio_loader['MagnificationFactor'], angio_loader['ImageSpacing'])
                #print('coord in mm ',pred_cord_mm,gt_coords_mm)
                if pred_cord_mm['x'] == [] and pred_cord_mm['y'] == []:
                    dict['Distance'].append(
                        str("Can't calculate Distance For this frame ( No prediction )"))
                else:
                    distance = calculate_distance(gt_coords_mm, pred_cord_mm)
                    dict['Distance'].append(distance)

                dict['Dice_forground'].append(dice_score[1].item())
                dict['Dice_background'].append(dice_score[0].item())
                dict['Patient'].append(patient)
                dict['Acquistion'].append(acquisition)
                dict['Frame'].append(frame)
                dict['Header'].append(angio_loader)
                dict['ImageSpacing'].append(angio_loader['ImageSpacing'])
                dict['MagnificationFactor'].append(
                    angio_loader['MagnificationFactor'])
                dict['Ann'].append(ann[f'{frame}'])

                # dict['Distance'].append()

            pbar.update(ins.shape[0])

        dice_score = metric.compute()

        print(f'[INFO] Test Dice score is {dice_score[1]*100:.2f} %')
        # dataframe['MSE']=mse_column['MSE']

        return dict


def main():

    config = None
    with open('config.yaml') as f:  # reads .yml/.yaml files
        config = yaml.safe_load(f)

    yml_data = yaml.dump(config)
    directory = f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
    parent_dir = config['data']['parent_dir_exp']
    path = pt.Path(parent_dir)/directory
    path.mkdir(exist_ok=True)
    dir = r'Predictii_Overlap'

    dir_2 = r'Gif_prediction_overlap'
    gif_path = pt.Path(path)/dir_2
    gif_path.mkdir(exist_ok=True)

    overlap_pred_path = pt.Path(path)/dir
    overlap_pred_path.mkdir(exist_ok=True)

    f = open(f"{path}\\yaml_config.yml", "w+")
    f.write(yml_data)
    f.close()
    path_construct = glob.glob(r"E:\__RCA_bif_detection\data\*")

    # path_list=create_dataset_csv(path_construct)
    #dataset_df = pd.DataFrame(path_list)

    #dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
    #print (dataset_df.head(3))

    dataset_df = pd.read_csv(config['data']['dataset_csv'])
    test_df = dataset_df.loc[dataset_df["subset"] == "test", :]
    test_ds = AngioClass(test_df, img_size=config["data"]["img_size"])
    # print(test_ds[0])

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["train"]["bs"], shuffle=False)

    network = torch.load(config['data']['model'])

    # print(f"# Test: {len(test_ds)}")

    test_set_CSV = test(network, test_loader, test_df,
                        thresh=config['test']['threshold'])
    dataf = pd.DataFrame(test_set_CSV)

    dice_histogram_maker(
        test_set_CSV['Dice_forground'], test_set_CSV['Dice_background'], path)
    mean_dice = []
    mean_distance = []

    patients = dataf['Patient'].unique()
    print(patients)
    for patient in patients:
        dataf_patient = dataf.loc[dataf["Patient"] == patient]
        print(dataf_patient)
        sum_dice = 0
        sum_distance = 0
        for index in dataf_patient.index:
            sum_dice += dataf_patient['Dice_forground'][index]
            print(len(dataf_patient['Distance'][index]))
            if dataf_patient['Distance'][index] == "Can't calculate Distance For this frame ( No prediction )":
                sum_distance += -1

            else:
                if len(dataf_patient['Distance'][index]) > 1:
                    for distance in dataf_patient['Distance'][index]:
                        print(sum_distance)
                        sum_distance = sum_distance + distance
                else:
                    sum_distance += dataf_patient['Distance'][index][0]

        print(type(dataf_patient), len(dataf_patient))
        print(sum_dice, sum_distance)
        m_dice = (sum_dice)/(len(dataf_patient.index))
        m_distance = sum_distance/(len(dataf_patient.index))
        for i in range(len(dataf_patient.index)):
            mean_dice.append(m_dice)
            mean_distance.append(m_distance)

    dataf["MeanDice"] = mean_dice
    dataf["MeanDIstance"] = mean_distance

    overlap_path = []
    prediction_path = []
    for batch_index, batch in enumerate(test_loader):
        x, y, index = iter(batch)

        index = index.numpy()

        network.eval()
        x = x.type(torch.cuda.FloatTensor)
        print(x.shape)
        y_pred = network(x.to(device='cuda:0'))
       # print( y_pred.shape)
       # print (len(x),x.shape)

        for step, (input, gt, pred) in enumerate(zip(x, y, y_pred)):
            #print (pred.shape, img.shape,gt.shape)
            # print(pred.shape)
            pred = F.softmax(pred, dim=0)[1].detach().cpu().numpy()
            # print(pred.shape, pred.min(), pred.max())
            pred[pred > config['test']['threshold']] = 1
            pred[pred <= config['test']['threshold']] = 0

            np_input = input.cpu().detach().numpy()
            np_gt = gt.cpu().detach().numpy()

            #print (pred.shape,input.shape)
            #print(np_gt.dtype,np_gt.shape, np_gt.min(),np_gt.max())
            # dst=overlap(np_input[0],pred)
            overlap_colors = overlap_3_chanels(np_gt[0], pred, np_input[0])

            # plt.axis('off')
            # plt.title('Prediction ')
            # plt.imshow(pred, cmap='gray')
            # plt.savefig(f"{path}\\Măști_frame{batch_index}_{step}.png")

            # plt.title('Overlap ')
            # plt.imshow(np_input[0],cmap='gray',interpolation=None)
            # plt.imshow(pred,cmap='jet',interpolation=None,alpha=0.8)
            # plt.savefig(f"{path}\\OVERLAP_frame{batch_index}_{step}.png")

            #cv2.imwrite(os.path.join(path, 'OVERLAP'+'_'+str(batch_index)+'_'+str(step)+'.png'),dst)
            overlap_path.append(os.path.join(overlap_pred_path, 'OVERLAP_Colored'+'_'+str(test_set_CSV['Patient'][index[step]])+'_'+str(
                test_set_CSV['Acquistion'][index[step]])+'-'+str(test_set_CSV['Frame'][index[step]])+'.png'))
            prediction_path.append(os.path.join(overlap_pred_path, 'PREDICTIE'+'_'+str(test_set_CSV['Patient'][index[step]])+'_'+str(
                test_set_CSV['Acquistion'][index[step]])+'-'+str(test_set_CSV['Frame'][index[step]])+'.png'))
            cv2.imwrite(os.path.join(overlap_pred_path, 'OVERLAP_Colored'+'_'+str(test_set_CSV['Patient'][index[step]])+'_'+str(
                test_set_CSV['Acquistion'][index[step]])+'-'+str(test_set_CSV['Frame'][index[step]])+'.png'), overlap_colors)
            cv2.imwrite(os.path.join(overlap_pred_path, 'PREDICTIE'+'_'+str(test_set_CSV['Patient'][index[step]])+'_'+str(
                test_set_CSV['Acquistion'][index[step]])+'-'+str(test_set_CSV['Frame'][index[step]])+'.png'), pred*255)

    dataf["OVERLAP_path"] = overlap_path
    dataf["PREDICTIE_path"] = prediction_path

    acquistions = dataf['Acquistion'].unique()
    print(patients)
    for acquistion in acquistions:
        dataf_ac = dataf.loc[dataf["Acquistion"] == acquistion]
        print(dataf_ac)
        movie_overlap_gif = []
        movie_predictie_gif = []
        for index in dataf_ac.index:
            dice_forground = dataf_ac['Dice_forground'][index]
            distance = dataf_ac['Distance'][index]
            frame = imageio.imread(dataf_ac['OVERLAP_path'][index])
            pred = imageio.imread(dataf_ac['PREDICTIE_path'][index])
            foo_Overlap = cv2.putText(
                frame, 'Dice:', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            foo_Overlap = cv2.putText(
                frame, f'{dice_forground:.2f}', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            foo_pred = cv2.putText(
                frame, 'dice:', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            foo_pred = cv2.putText(
                frame, f'{dice_forground:.2f}', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            foo_Overlap = cv2.putText(
                foo_Overlap, 'Distance:', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            print (distance)
            if (distance == "Can't calculate Distance For this frame ( No prediction )") or (distance[0] == "Can't calculate Distance For this frame ( No prediction )"):
                foo_Overlap = cv2.putText(
                    foo_Overlap, f'{distance}', (5, 65), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            else:
                
                dist =[]
                for d in distance :
                    d=float (d)
                    d=round(d,2)
                    print (d)
                    dist.append(d)

                distance=dist
                
            foo_Overlap = cv2.putText(
                foo_Overlap, f'{distance}', (5, 65), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            
            
            if (distance == "Can't calculate Distance For this frame ( No prediction )") or (distance[0] == "Can't calculate Distance For this frame ( No prediction )"):
                 foo_pred = cv2.putText(
                    foo_pred, f'{distance}', (5, 65), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            else:
               
                for d in distance :
                    d=float (d)
                    d=round(d)
            foo_pred = cv2.putText(
                foo_pred, f'{distance}', (5, 65), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
           
            movie_overlap_gif.append(foo_Overlap)
            movie_predictie_gif.append(foo_pred)

        imageio.mimsave(os.path.join(gif_path, 'OVERLAP_GIF'+'_' +
                        str(acquistion)+'.gif'), movie_overlap_gif, duration=1)
        imageio.mimsave(os.path.join(gif_path, 'PREDICTIE_GIF'+'_' +
                        str(acquistion)+'.gif'), movie_predictie_gif, duration=1)

    histogram_distance(dataf,path)
    dataf.to_csv(f"{path}\\CSV_TEST.csv")


if __name__ == "__main__":
    main()
