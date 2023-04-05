from json.tool import main
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchmetrics
import yaml
import torch
import torch.nn as nn
from datetime import datetime
import pandas as pd
import pathlib as pt
import yaml
import torch.nn as nn
import monai.transforms as TR
import torchmetrics
from tqdm import tqdm
from datetime import datetime
import sys
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt
from comet_ml import Experiment
from regresie import RegersionClass
import albumentations as A 

def plot_acc_loss(result, path):
    acc = result['MSE']['train']
    loss = result['loss']['train']
    val_acc = result['MSE']['valid']
    val_loss = result['loss']['valid']

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('MSE', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('MSE')
    plt.xlabel('Epoch')

    plt.subplot(122)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig(f"{path}/Curbe de învățare")


def set_parameter_requires_grad(model, freeze):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False


def train(network, train_loader, valid_loader, exp, criterion, opt, epochs, thresh=0.5, weights_dir='weights', save_every_ep=50):

    total_loss = {'train': [], 'valid': []}
    total_dice = {'train': [], 'valid': []}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting training on device {device} ...")

    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    metric = MeanSquaredError()
    network.to(device)
    criterion.to(device)

    for ep in range(epochs):

        print(f"[INFO] Epoch {ep}/{epochs - 1}")

        print("-" * 20)
        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_average = 0.0

            if phase == 'train':
                network.train()  # Set model to training mode
            else:
                network.eval()   # Set model to evaluate mode

            with tqdm(desc=phase, unit=' batch', total=len(loaders[phase].dataset)) as pbar:
                for data in loaders[phase]:
                    ins, tgs, idx = data
                    # print(data)
                    #print(type(ins), type(tgs))
                    ins = ins.to(device)
                    tgs = tgs.to(device)
                    #print (ins.size(),tgs.size())

                    # seteaza toti gradientii la zero, deoarece PyTorch acumuleaza valorile lor dupa mai multe backward passes
                    opt.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # se face forward propagation -> se calculeaza predictia
                        # plt.imshow(ins[0][0].cpu(),cmap='gray')
                        # plt.show()
                        
                        output = network(ins)

                        # plt.imshow(output[0][0].cpu().detach().numpy(),cmap='gray')
                        # plt.show()
                        # print(output.size())

                        #second_output = Variable(torch.argmax(output,1).float(),requires_grad=True).cuda()
                        # output[:, 1, :, :] #=> 8 x 128 x 128
                        # tgs => 8 x 1 x 128 x 128
                        # tgs.squeeze() #=> 8 x 128 x 128
                        #print('Output ', output.shape , output.dtype , output.min(), output.max())
                        #print('target',tgs.shape , tgs.dtype , tgs.min(), tgs.max())
                        # se calculeaza eroarea/loss-ul
                        print (output.shape , tgs.squeeze())
                        loss = criterion(output, tgs.squeeze())

                        #plt.imshow(current_predict[0].cpu().detach().numpy(), cmap='gray')
                        # plt.show()

                        if 'cuda' in device.type:
                            output = output.cpu()
                            tgs = tgs.cpu().type(torch.int).squeeze()
                        else:
                            tgs = tgs.type(torch.int).squeeze()

                        # plt.imshow(current_target[0].cpu().detach().numpy())
                        # plt.show()

                        #print(current_predict.shape, current_target.shape)
                        #print(current_predict.dtype, current_target.dtype
                        mse = metric(output, tgs)
                        # print(dice_idx.item)

                        #print(dice_idx, dice_idx.shape, dice_idx.dtype )
                        # print(current_predict,current_target)
                        # print(f"\tAcc on batch {i}: {acc}")

                        if phase == 'train':
                            # se face backpropagation -> se calculeaza gradientii
                            loss.backward()
                            # se actualizează weights-urile
                            opt.step()

                    running_loss += loss.item() * ins.size(0)

                    running_average += mse.item() * ins.size(0)

                    #print(running_average, dice_idx.item())
                    #print(running_loss, loss.item())

                    if phase == 'valid':
                        # salvam ponderile modelului dupa fiecare epoca
                        if ep % save_every_ep == 0:
                            torch.save(
                                network, f"{weights_dir}/my_model{datetime.now().strftime('%m%d%Y_%H%M')}_e{ep}.pt")

                    #     model_path = f"{weights_dir}\\model_epoch{ep}.pth"
                    #     torch.save({'epoch': ep,
                    #                 'model_state_dict': network.state_dict(),
                    #                 'optimizer_state_dict': opt.state_dict(),
                    #                 'loss': total_loss,
                    #                 }, model_path)

                    pbar.update(ins.shape[0])

                # Calculam loss-ul pt toate batch-urile dintr-o epoca
                total_loss[phase].append(
                    running_loss/len(loaders[phase].dataset))

                loss_value = running_loss/len(loaders[phase].dataset)
                mse_value = running_average/len(loaders[phase].dataset)
                # Calculam acuratetea pt toate batch-urile dintr-o epoca
                total_dice[phase].append(
                    running_average/len(loaders[phase].dataset))

                postfix = f'error {total_loss[phase][-1]:.4f} MSE {mse*100:.2f}%'
                pbar.set_postfix_str(postfix)

                exp.log_metrics({f"{phase}MSE": mse_value,
                                f"{phase}loss": loss_value}, epoch=ep)

                # Resetam pt a acumula valorile dintr-o noua epoca

    return {'loss': total_loss, 'MSE': total_dice}


def main():

    print(f"pyTorch version {torch.__version__}")
    print(f"torchvision version {torchvision.__version__}")
    print(f"torchmetrics version {torchmetrics.__version__}")
    print(f"CUDA available {torch.cuda.is_available()}")

    config = None
    with open('config.yaml') as f:  # reads .yml/.yaml files
        config = yaml.safe_load(f)

    experiment = Experiment(
        api_key="wwQKu3dl9l1bRZOpeKs0y3r8S",
        project_name="general",
        workspace="razvanavesalon",)

    exp_name = f"Experiment_MSE{datetime.now().strftime('%m%d%Y_%H%M')}"

    path_1=pt.Path.cwd()
    exp_path = path_1/exp_name  # get current working directory
    exp_path.mkdir(exist_ok=True)
    dir = "Weights"
    path = pt.Path(exp_path)/dir
    path.mkdir(exist_ok=True)
    #network = UNet(n_channels=1, n_classes=2,final_activation=nn.Sigmoid())

  
    experiment.log_parameters(config)

    yml_data = yaml.dump(config)
    f = open(f"{path}/yaml_config.yml", "w+")
    f.write(yml_data)
    f.close()

    pixels = T.Compose([

        TR.ToTensord(keys="img"),
    ])

    # pixel_t = TR.Compose([
    #     TR.GaussianSmoothd(keys="img", sigma=config['train']['sigma']),
    #     TR.RandGibbsNoised(
    #         keys="img", prob=config['train']['gibbs_noise_prob'], alpha=config['train']['alpha']),
    #     TR.RandAdjustContrastd(
    #         keys="img", prob=config['train']['contrast_prob'], gamma=config['train']['contrast_gamma']),
    # ])
    
    pixel_t=A.Compose([
        A.CLAHE(clip_limit=config['train']['clip_limit'], tile_grid_size=config['train']['tile_grid_size'], always_apply=False, p=config['train']['p_clahe']),
        A.GaussianBlur(blur_limit=config['train']['blur_limit'], sigma_limit=config['train']['sigma_limit'], always_apply=False, p=config['train']['p_gauss_blur']),
        A.RandomGamma(gamma_limit=config['train']['gamma_limit'], eps=None, always_apply=False, p=config['train']['p']),
        
        ],keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))


    geometric_t= A.Compose([
        A.Rotate(limit=config['train']['rotate_range']),
        A.Resize(height=config['data']['img_size'][0] , width=config['data']['img_size'][1])
    ],keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))
    
    # geometric_t = TR.Compose([

    #     TR.RandRotated(keys=["img", "seg"], prob=config['train']['rotate_prob'],
    #                    range_x=config['train']['rotate_range'], mode=['bilinear', 'nearest']),
    #     TR.RandFlipd(keys=["img", "seg"], prob=config['train']['flip_prob'],
    #                  spatial_axis=config['train']['flip_spatial_axis']),
    #     TR.RandZoomd(keys=["img", "seg"], prob=config['train']['zoom_prob'],
    #                  min_zoom=config['train']['min_zoom'], max_zoom=config['train']['max_zoom'])
    #     #TR.RandSpatialCropSamplesd(keys=["img", "seg"],num_samples=config['train']['rand_crop_samples'], roi_size=config['train']['rand_crop_size'],random_size=False),
    # ])

    #path_construct = glob.glob(config["data"]['data_path'])
    #path_list = create_dataset_csv(path_construct)
    #dataset_df = pd.DataFrame(path_list)

    #dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
    # print(dataset_df.head(3))
    # dataset_df.to_csv(config['data']['dataset_csv'])

    dataset_df = pd.read_csv(config['data']['dataset_csv'])

    train_df = dataset_df.loc[dataset_df["subset"] == "train",:]
    print(train_df)
    train_ds = RegersionClass(train_df, img_size=config['data']['img_size'],pixel_transforms=pixel_t,geometrics_transforms=geometric_t)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config['train']['bs'], shuffle=True,drop_last=True)

    valid_df = dataset_df.loc[dataset_df["subset"] == "valid", :]
    print(valid_df)
    valid_ds = RegersionClass(valid_df, img_size=config['data']['img_size'],pixel_transforms=pixels)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=config['train']['bs'], shuffle=False,drop_last=True)

    print(f"# Train: {len(train_ds)} # Valid: {len(valid_ds)}")

  # Specificarea functiei loss
    criterion = nn.MSELoss()
    n_classes = 2
    # network = torchvision.models.resnet34(pretrained=True)
    # set_parameter_requires_grad(network, freeze=False)
    # num_ftrs = network.fc.in_features
    # network.fc = nn.Linear(num_ftrs, n_classes)
    # print(network)
    
    #
    network=torchvision.models.efficientnet_b1(pretrained=True)
    num_ftrs = network.classifier[1].in_features
    network.classifier[1]=nn.Linear(num_ftrs, n_classes)
    #CustomNet(1, 16, 32 ,64, 2)
    # definirea optimizatorului

    if config['train']['opt'] == 'Adam':
        opt = torch.optim.Adam(network.parameters(), lr=config['train']['lr'])
    elif config['train']['opt'] == 'SGD':
        opt = torch.optim.SGD(network.parameters(), lr=config['train']['lr'])
    elif config['train']['opt'] == "RMSprop":
        opt = torch.optim.RMSprop(
            network.parameters(), lr=config['train']['lr'])

    history = train(network, train_loader, valid_loader, experiment, criterion, opt,
                    epochs=config['train']['epochs'], thresh=config['test']['threshold'], weights_dir=path)
    plot_acc_loss(history, path)


if __name__ == "__main__":
    main()

