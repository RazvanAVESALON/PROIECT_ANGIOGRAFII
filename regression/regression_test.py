import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import yaml
import cv2
import torch
from tqdm import tqdm
from datetime import datetime
import os
import cv2
import glob
from torchmetrics import MeanSquaredError
import skimage.color
import json
import imageio
from regresie import RegersionClass
from distances_regression import calculate_distance,mm2pixels,pixels2mm
import albumentations as A
from acq_hist_maker import hist_metrics_per_acq
# def (dataset_df):

#     for i in range()


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
    print(input.shape)
    pred = pred.astype(np.int32)
    gt = gt.astype(np.int32)

    tp = gt & pred
    fp = ~gt & pred
    fn = gt & ~pred
    tn = ~gt & ~pred

    print(tp.min(), tp.max(), fp.min(), fp.max(), fn.min(), fn.max())

    img = np.zeros((256, 256, 3), np.float32)
    img[:, :, 1] = tp
    img[:, :, 2] = fp
    img[:, :, 0] = fn

    input = skimage.color.gray2rgb(input)
    print(img.min(), img.max(), img.dtype, img.shape)
    print(input.min(), input.max(), input.dtype, input.shape)
    dst = cv2.addWeighted(input, 0.7, img, 0.3, 0)
    #dst = cv2.resize(dst,dsize,interpolation=cv2.INTER_AREA)
    # plt.imshow(img)
    # plt.show()

    return dst


def mse_histogram_maker(mse, path):
    plt.hist(mse, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Frames per Image')
    plt.xlabel('RMSE Values per interval')
    plt.ylabel("Count of frames per rmse value ")
    plt.savefig(f"{path}\\Histograma_rmse")


def test(network, test_loader, dataframe, thresh=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting testing on device {device} ...")

    metric = MeanSquaredError()

    logs_dict = {'MSE': [], 'Patient': [], 'Acquistion': [], 'Frame': [], 'Distance':[]}

    # mse_column={'MSE':[]}
    network.eval()
    with tqdm(desc='test', unit=' batch', total=len(test_loader.dataset)) as pbar:

        #print (test_loader.dataset[100])
        for data in test_loader:

            ins, tgs, index = data
            print(index)

            network.to(device)
            ins = ins.to(device)
            metric = metric.to(device)
            tgs = tgs.to(device)
            output = network(ins)
            print(output, tgs)

            print(output)
           # if 'cuda' in device.type:
            # tgs = tgs.cpu()
            # output=output.cpu()

            #print (current_predict.shape, tgs.shape)
            for batch_idx, (frame_pred, target_pred) in enumerate(zip(output, tgs)):
                print(frame_pred)
                #frame_pred=frame_pred.int()
                MSE_score = metric(frame_pred, target_pred)

                print('MSE', MSE_score)
                patient, acquisition, frame, header, annotations = test_loader.dataset.csvdata(
                    (index[batch_idx].numpy()))
                # print(header)

                with open(annotations) as g:
                    ann = json.load(g)

                with open(header) as f:
                    angio_loader = json.load(f)
                print('frame_pred',frame_pred)
                frame_pred=frame_pred.cpu().detach().numpy()
                target_pred=target_pred.cpu().detach().numpy()
                print('frame_pred',frame_pred)
                gt_coords_mm=pixels2mm(target_pred,angio_loader['MagnificationFactor'],angio_loader['ImageSpacing'])
                pred_cord_mm=pixels2mm(frame_pred,angio_loader['MagnificationFactor'],angio_loader['ImageSpacing'])
                print('coord in mm ',pred_cord_mm,gt_coords_mm)
                if not len(pred_cord_mm) and not len(pred_cord_mm):
                    logs_dict['Distance'].append( str("Can't calculate Distance For this frame ( No prediction )" ) )
                else:
                    distance=calculate_distance(gt_coords_mm,pred_cord_mm)
                  
                    logs_dict['Distance'].append(distance)

                logs_dict['MSE'].append(MSE_score.cpu().detach().numpy())
                logs_dict['Patient'].append(patient)
                logs_dict['Acquistion'].append(acquisition)
                logs_dict['Frame'].append(frame)

            pbar.update(ins.shape[0])

        MSE_score = metric.compute()
        print(MSE_score)

        print(f'[INFO] MSE score is {MSE_score:.2f} %')
        # dataframe['MSE']=mse_column['MSE']

        return logs_dict


def main():

    config = None
    with open('config.yaml') as f:  # reads .yml/.yaml files
        config = yaml.safe_load(f)

    yml_data = yaml.dump(config)
    directory = f"Test{datetime.now().strftime('%m%d%Y_%H%M')}"
    parent_dir = r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\experiments\Experimente Regresie\Experiment_MSE02112023_0136"
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
    resize= A.Compose([
        A.Resize(height=config['data']['img_size'][0] , width=config['data']['img_size'][1])
    ],keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))
    
    dataset_df = pd.read_csv(config['data']['dataset_csv'])
    test_df = dataset_df.loc[dataset_df["subset"] == "test", :]
    test_ds = RegersionClass(test_df, img_size=config["data"]["img_size"],geometrics_transforms=resize)
    # print(test_ds[0])

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["train"]["bs"], shuffle=False)

    network = torch.load(
        r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\experiments\Experimente Regresie\Experiment_MSE02112023_0136\Weights\my_model02122023_0037_e100.pt")

    # print(f"# Test: {len(test_ds)}")

    test_set_CSV = test(network, test_loader, test_df,
                        thresh=config['test']['threshold'])
    dataf = pd.DataFrame(test_set_CSV)

    mse_histogram_maker(test_set_CSV['MSE'], path)
    mean_MSE = []
    mean_distance = []

    patients = dataf['Patient'].unique()
    print(patients)
    for patient in patients:
        dataf_patient = dataf.loc[dataf["Patient"] == patient]
        print(dataf_patient)
        sum_mse = 0
        sum_distance = 0
        for index in dataf_patient.index:
            sum_mse += dataf_patient['MSE'][index]
            if dataf_patient['Distance'][index] == "Can't calculate Distance For this frame ( No prediction )":
                sum_distance += -1

            else:
                sum_distance += dataf_patient['Distance'][index]
                   

        print(type(dataf_patient), len(dataf_patient))
        print(sum_mse, sum_distance)
        m_dice = sum_mse/(len(dataf_patient.index))
        m_distance = sum_distance/(len(dataf_patient.index))
        for i in range(len(dataf_patient.index)):
            mean_MSE.append(m_dice)
            mean_distance.append(m_distance)

    dataf["MeanMSE"] = mean_MSE
    dataf["MeanDIstance"] = mean_distance

    hist_metrics_per_acq(dataf,path)
    
    overlap_path = []
    prediction_path = []
    for batch_index, batch in enumerate(test_loader):
        x, y, index = iter(batch)

        index = index.numpy()

        network.eval()
        x = x.type(torch.cuda.FloatTensor)

        y_pred = network(x.to(device='cuda:0'))
       # print( y_pred.shape)
       # print (len(x),x.shape)

        for step, (input, gt, pred) in enumerate(zip(x, y, y_pred)):
            #print (pred.shape, img.shape,gt.shape)
            # print(pred.shape)
            # print(pred.shape, pred.min(), pred.max())

            np_input = input.cpu().detach().numpy()*255
            gt = gt.cpu().detach().numpy()*255
            print("Inputul  ESTE ", np_input.shape)
            pred = pred.cpu().detach().numpy()*255

            black = np.zeros(np_input.shape[1:3])
            masked_gt = cv2.circle(
                black, (int(gt[1]), int(gt[0])), 5, [255, 255, 255], -1)
            black2 = np.zeros(np_input.shape[1:3])
            masked_pred = cv2.circle(
                black2, (int(pred[1]), int(pred[0])), 5, [255, 255, 255], -1)

            #print (pred.shape,input.shape)
            #print(np_gt.dtype,np_gt.shape, np_gt.min(),np_gt.max())
            # dst=overlap(np_input[0],pred)
            overlap_colors = overlap_3_chanels(
                masked_gt, masked_pred, np_input[0])

            # plt.axis('off')
            # plt.title('Prediction ')
            # plt.imshow(pred, cmap='gray')
            # plt.savefig(f"{path}\\Măști_frame{batch_index}_{step}.png")

            # plt.title('Overlap ')
            # plt.imshow(np_input[0],cmap='gray',interpolation=None)
            # plt.imshow(pred,cmap='jet',interpolation=None,alpha=0.8)
            # plt.savefig(f"{path}\\OVERLAP_frame{batch_index}_{step}.png")

            #cv2.imwrite(os.path.join(path, 'OVERLAP'+'_'+str(batch_index)+'_'+str(step)+'.png'),dst)
            frame = str(test_set_CSV['Frame'][index[step]])
            pat_id = str(test_set_CSV['Patient'][index[step]])
            acn_id = str(test_set_CSV['Acquistion'][index[step]])
            
            curr_overlap_path = str(overlap_pred_path/f"OVERLAP_Colored_{pat_id}_{acn_id}-{frame}.png")
            curr_prediction_path = str(overlap_pred_path/f"PREDICTIE{pat_id}_{acn_id}-{frame}.png")
            overlap_path.append(curr_overlap_path)
            prediction_path.append(curr_prediction_path)
            
            cv2.imwrite(curr_overlap_path, overlap_colors)
            cv2.imwrite(curr_prediction_path, pred*255)

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
            mse = dataf_ac['MSE'][index]
            distance = dataf_ac['Distance'][index]
            frame = imageio.imread(dataf_ac['OVERLAP_path'][index])
            pred = imageio.imread(dataf_ac['PREDICTIE_path'][index])
            foo_Overlap = cv2.putText(
                frame, 'MSE:', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            foo_Overlap = cv2.putText(
                frame, f'{mse:.2f}', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            foo_pred = cv2.putText(
                frame, 'MSE:', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            foo_pred = cv2.putText(
                frame, f'{mse:.2f}', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            foo_Overlap = cv2.putText(
                foo_Overlap, 'Distance:', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            print (distance)
            foo_Overlap = cv2.putText(foo_Overlap, f'{distance:.2f}', (5, 65), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))
            foo_pred = cv2.putText(foo_pred, f'{distance:.2f}', (5, 65), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))

           
            movie_overlap_gif.append(foo_Overlap)
            movie_predictie_gif.append(foo_pred)

        imageio.mimsave(os.path.join(gif_path, 'OVERLAP_GIF'+'_' +
                        str(acquistion)+'.gif'), movie_overlap_gif, duration=1)
        imageio.mimsave(os.path.join(gif_path, 'PREDICTIE_GIF'+'_' +
                        str(acquistion)+'.gif'), movie_predictie_gif, duration=1)


    dataf.to_csv(f"{path}\\CSV_TEST.csv")


if __name__ == "__main__":
    main()
