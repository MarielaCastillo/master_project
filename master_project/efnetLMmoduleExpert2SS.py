import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset

import segmentation_models_pytorch as smp

import pandas as pd
import os
from skimage import io

import numpy as np

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


class MultiModalDataset(Dataset):
    def __init__(self, rgb_path, thermo_path, label_path, transform=None):
        self.rgb_path = rgb_path
        self.thermo_path = thermo_path
        self.label_path = label_path
        self.rgb_filepaths = [f for f in sorted(os.listdir(rgb_path))]
        self.thermo_filepaths = [f for f in sorted(os.listdir(thermo_path))]
        self.label_filepaths = [f for f in sorted(os.listdir(label_path))]
        self.transform = transform

    def __getitem__(self, idx):
        img_rgb = io.imread(self.rgb_path+'/'+self.rgb_filepaths[idx])
        img_thermo = io.imread(self.thermo_path+'/'+self.thermo_filepaths[idx])
        label = io.imread(self.label_path+'/'+self.label_filepaths[idx])

        img_thermo = img_thermo.astype(float)

        if self.transform:
            img_thermo = self.transform(img_thermo)
            img_rgb = self.transform(img_rgb)
            label = self.transform(label)
            
        #return img_rgb, img_thermo, label
        return img_thermo, label
        #return img_rgb, label

    def __len__(self):
        return len(self.rgb_filepaths)

class UNetExpert1(nn.Module):
    def __init__(self, inchannels, numclasses):
        super().__init__()
        self.model = smp.Unet('efficientnet-b4', in_channels=inchannels, classes=numclasses, activation='softmax').double()
        #self.features = salida del encoder
        #self.decoder = asdf
    
    def forward(self, input):
        #hidden_state = 1raparte(input)
        #output = self.decoder(hidden_state)
        hidden_state = self.model.encoder(input)
        decoder_output = self.model.decoder(*hidden_state)

        masks = self.model.segmentation_head(decoder_output)
        '''
        if self.model.classification_head is not None:
            labels = self.model.classification_head(hidden_state[-1])
            return hidden_state, masks, labels
        '''
        return hidden_state, masks

   
####Model
class LitModelEfficientNet(pl.LightningModule):
    #https://github.com/gthparch/edgeBench/blob/master/pytorch/models/cifarnet.py#L23
    def __init__(self, batch_size, transform):
        super(LitModelEfficientNet, self).__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.MSELoss()

        self.cnnexpertThermo = UNetExpert1(inchannels=1, numclasses=13)

        #self.cnnexpertRGB = smp.Unet('resnet34', classes=13, activation='softmax')
        #self.cnnexpertRGB = CNNExpert1(3, 3) #model_weights[key.replace("auto_encoder.", "")] = model_weights.pop(key)

    def forward(self, x1): ### here1
        #hidden_statergb, outrgb = self.cnnexpertRGB(x1)

        #output = self.cnnexpertThermo(x1)
        #output, _ = self.cnnexpertThermo(x1)
        _, output = self.cnnexpertThermo(x1) #why
        return output

    def train_dataloader(self):
        print("dir_path ", dir_path)
        print(dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00/fl_rgb/')
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00'

        trainset = MultiModalDataset(rgb_path= dir_path2 + '/' + 'fl_rgb', 
                                        thermo_path = dir_path2 + '/' + 'fl_ir_aligned',
                                        label_path= dir_path2 + '/' + 'fl_rgb_labels',
                                        transform= self.transform)  

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=10)                                                                             
        return trainloader

    def test_dataloader(self):
        print("dir_path ", dir_path)
        print(dir_path + '/' + 'thermaldatasetfolder/test/seq_01_day/00/fl_rgb/')
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/test/seq_01_day/00'

        testset = MultiModalDataset(rgb_path= dir_path2 + '/' + 'fl_rgb', 
                                        thermo_path = dir_path2 + '/' + 'fl_ir_aligned',
                                        label_path= dir_path2 + '/' + 'fl_rgb_labels',
                                        transform= self.transform)  
                                        
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=10)   
        return testloader

    ### optimizers and schedulers
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        inputthermo, labels = train_batch
        #optimizer.zero_grad()

        outputs = self(inputthermo) #forward(x1, x2, x3, x4)
        
        labels = labels.repeat(1, 3, 1, 1) #check this

        loss = self.criterion(outputs, labels)
        return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        outputs = self(images)
    
        _, predicted = torch.max(outputs.data, 1) ## why this?

        loss = F.cross_entropy(predicted, labels)
        #loss = F.mse_loss(predicted, labels)

        return loss