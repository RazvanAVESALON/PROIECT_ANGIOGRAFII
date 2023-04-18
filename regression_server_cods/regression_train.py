from json.tool import main
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchmetrics
import yaml
import torch
import torch.nn as nn
from datetime import datetime
import pandas as pd
import pathlib as pt
import yaml
import torch.nn as nn
import torchmetrics
from tqdm import tqdm
from datetime import datetime
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
    total_mse = {'train': [], 'valid': []}

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
            running_mse = 0.0

            if phase == 'train':
                network.train()
            else:
                network.eval()
            with tqdm(desc=phase, unit=' batch', total=len(loaders[phase].dataset)) as pbar:
                for data in loaders[phase]:
                    ins, tgs, idx = data

                    ins = ins.to(device)
                    tgs = tgs.to(device)

                    opt.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):

                        output = network(ins)

                        loss = criterion(output, tgs.squeeze())

                        if 'cuda' in device.type:
                            output = output.cpu()
                            tgs = tgs.cpu().type(torch.int).squeeze()
                        else:
                            tgs = tgs.type(torch.int).squeeze()

                        mse = metric(output, tgs)

                        if phase == 'train':
                            loss.backward()
                            opt.step()

                    running_loss += loss.item() * ins.size(0)

                    running_mse += mse.item() * ins.size(0)

                    if phase == 'valid':

                        if ep % save_every_ep == 0:
                            torch.save(
                                network, f"{weights_dir}/my_model{datetime.now().strftime('%m%d%Y_%H%M')}_e{ep}.pt")

                    pbar.update(ins.shape[0])
                total_loss[phase].append(
                    running_loss/len(loaders[phase].dataset))
                total_mse[phase].append(
                    running_mse/len(loaders[phase].dataset))

                postfix = f'error {total_loss[phase][-1]:.4f} MSE {mse*100:.2f}%'
                pbar.set_postfix_str(postfix)

                exp.log_metrics({f"{phase}MSE": total_mse[phase][-1],
                                f"{phase}loss": total_loss[phase][-1]}, epoch=ep)

    return {'loss': total_loss, 'MSE': total_mse}


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

    path_1 = pt.Path(
        '/media/cuda/HDD 1TB  - DATE/AvesalonRazvanDate , Experimente/Experimente')
    exp_path = path_1/exp_name
    exp_path.mkdir(exist_ok=True)
    dir = "Weights"
    path = pt.Path(exp_path)/dir
    path.mkdir(exist_ok=True)

    experiment.log_parameters(config)

    yml_data = yaml.dump(config)
    f = open(f"{path}/yaml_config.yml", "w+")
    f.write(yml_data)
    f.close()

    pixel_t = A.Compose([
        A.CLAHE(clip_limit=config['train']['clip_limit'], tile_grid_size=config['train']
                ['tile_grid_size'], always_apply=False, p=config['train']['p_clahe']),
        A.GaussianBlur(blur_limit=config['train']['blur_limit'], sigma_limit=config['train']
                       ['sigma_limit'], always_apply=False, p=config['train']['p_gauss_blur']),
        A.RandomGamma(gamma_limit=config['train']['gamma_limit'],
                      eps=None, always_apply=False, p=config['train']['p']),

    ])

    geometric_t = A.Compose([
        A.Rotate(limit=config['train']['rotate_range']),
        A.Resize(height=config['data']['img_size'][0],
                 width=config['data']['img_size'][1])
    ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))

    g_t = A.Compose([
        A.Resize(height=config['data']['img_size'][0],
                 width=config['data']['img_size'][1])
    ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))

    dataset_df = pd.read_csv(config['data']['dataset_csv'])

    train_df = dataset_df.loc[dataset_df["subset"] == "train"]
    train_ds = RegersionClass(
        train_df, img_size=config['data']['img_size'], pixel_transforms=pixel_t, geometrics_transforms=geometric_t)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config['train']['bs'], shuffle=True, drop_last=True)

    valid_df = dataset_df.loc[dataset_df["subset"] == "valid", :]
    valid_ds = RegersionClass(
        valid_df, img_size=config['data']['img_size'], geometrics_transforms=g_t)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=config['train']['bs'], shuffle=False, drop_last=True)

    print(f"# Train: {len(train_ds)} # Valid: {len(valid_ds)}")
    criterion = nn.MSELoss()
    n_classes = 2

    #network = torchvision.models.resnet18(pretrained=False)
    #set_parameter_requires_grad(network, freeze=False)
    #num_ftrs = network.fc.in_features
    #network.fc = nn.Linear(num_ftrs, n_classes)

    network = torchvision.models.efficientnet_b0(pretrained=False)
    num_ftrs = network.classifier[1].in_features
    network.classifier[1] = nn.Linear(num_ftrs, n_classes)
    print(network)

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
