import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from collections import OrderedDict

import pandas as pd
import os
from skimage import io

import numpy as np

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, dtype=str)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index): ##collate #should it be reading one at a time???
        img_path = os.path.join(self.root_dir, f'{self.annotations.iloc[index, 0]}_rgb.png')
        imagergb = io.imread(img_path)

        img_path = os.path.join(self.root_dir, f'{self.annotations.iloc[index, 0]}.png')
        imagedepth = io.imread(img_path)

        pointcloud_path = os.path.join(self.root_dir, f'{self.annotations.iloc[index, 0]}.txt')
        #print("pointcloud_path ", pointcloud_path)

        df = pd.read_csv(pointcloud_path, sep=" ", header=None)
        dflidar = df.drop(df.columns[[3]], axis=1)
        torch_tensor = torch.tensor(dflidar.values)
        #RuntimeError: stack expects each tensor to be equal size, but got [122455, 3] at entry 0 and [121266, 3] at entry 1
        #collate

        y_label = torch.tensor([float(self.annotations.iloc[index, 1]),float(self.annotations.iloc[index, 2]),float(self.annotations.iloc[index, 3])])

        if self.transform:
            imagergb = self.transform(imagergb)
            imagedepth = self.transform(imagedepth)

        return [imagergb, imagedepth, y_label]
        #return [imagergb, imagedepth, torch_tensor, y_label]

class CNNExpert1(nn.Module):
    def __init__(self, inchannel, numclasses):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b4', in_channels=inchannel)
        self.disable_gradients(self.efficient_net)
        self.e2h = nn.Linear(1792, 64)
        self.hidden_dim = 64
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=256),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=numclasses)
        )

    def disable_gradients(self, model):
        # update the pretrained model
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, input): ### 8, 3, 128, 128
        features = self.efficient_net.extract_features(input)
        hidden_state = self.efficient_net._avg_pooling(features)
        hidden_state = hidden_state.flatten(start_dim=1)
        hidden_state = self.e2h(hidden_state)
        return hidden_state, self.fc(hidden_state)
   
####Model
class LitModelEfficientNet(pl.LightningModule):
    #https://github.com/gthparch/edgeBench/blob/master/pytorch/models/cifarnet.py#L23
    def __init__(self, batch_size, transform):
        super(LitModelEfficientNet, self).__init__()
        self.batch_size = batch_size
        self.transform = transform
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()

        self.cnnexpertDepth = CNNExpert1(1, 3) 

    def forward(self, x1): ### here1
        hidden_statedepth, outdepth = self.cnnexpertDepth(x1)
        return outdepth

    def train_dataloader(self):
        trainset = MultiModalDataset(csv_file=dir_path + '/' + 'multimodal.csv', root_dir=dir_path + '/'+'multimodalfolder', transform=self.transform)


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=10) #2
        return trainloader

    def test_dataloader(self):
        #testset = torchvision.datasets.Kitti(root='./data', train=False,
        #                                      download=True, transform=self.transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=self.transform)


        #test setdatoenvivo = # no en train, pero en vivo.  1 rgb 1bw 1 point cloud
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=10) #2
        return testloader

    ### optimizers and schedulers
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        inputrgb, inputdepth, labels = train_batch
        #optimizer.zero_grad()

        outputs = self(inputdepth) #forward(x1, x2, x3, x4)
        loss = self.criterion(outputs, labels)
        return loss

        
        #histate, outputs = self(inputrgb) #forward(x1, x2, x3, x4)
        #loss = self.criterion(outputs, labels)
        #return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        outputs = self(images)
    
        _, predicted = torch.max(outputs.data, 1)

        #loss = F.cross_entropy(predicted, labels)
        loss = F.mse_loss(predicted, labels)

        return loss