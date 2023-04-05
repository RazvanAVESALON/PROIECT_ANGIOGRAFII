import torch
import torch.nn as nn
import torchvision
import yaml
import torchvision.models as models
print(f"pyTorch version {torch.__version__}")
print(f"torchvision version {torchvision.__version__}")
print(f"CUDA available {torch.cuda.is_available()}")

config = None
with open('config.yaml') as f:
    config = yaml.safe_load(f)


n1 = 128
n2 = 256
n3 = 512


class Resnet18(nn.Module):
    def __init__(self, n_classes):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.linear = nn.Linear(self.num_features, n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.num_features
        x = self.linear(x)
        x = self.sig(x)
        return x


class CustomNet(torch.nn.Module):
    def __init__(self, input_nc, n1, n2, n3, n_classes):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, n1, (5, 5), padding='same')
        self.conv2 = nn.Conv2d(n1, n2, (5, 5), padding='same')
        # 4*4 image dimension after 2 max_pooling
        self.linear1 = nn.Linear(n2*64*64, config["train"]["bs"])
        self.linear2 = nn.Linear(config["train"]["bs"], n_classes)
        self.max_pool = nn.MaxPool2d((2, 2))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.sig(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.sig(x)
        x = self.max_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.sig(x)
        x = self.linear2(x)

        return x
