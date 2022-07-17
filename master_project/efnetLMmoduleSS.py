from tabnanny import check
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from collections import OrderedDict

import segmentation_models_pytorch as smp

import pandas as pd
import os
from skimage import io

import numpy as np
from torch.utils.data.dataloader import default_collate

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

#fl_ir_aligned_1570722156_952177040
#fl_rgb_1570722156_952177040
#fl_rgb_labels_1570722156_952177040

#1570722156_952177040
#1570722156_952177040
#1570722156_952177040

#fl_ir_aligned_1570729614_4316866600
#fl_rgb_1570729614_4316866600

#1570729614_4316866600
#1570729614_4316866600

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
        return img_rgb, label

    def __len__(self):
        return len(self.rgb_filepaths)

class UNetExpert1(nn.Module):
    def __init__(self, inchannels, numclasses):
        super().__init__()
        self.model = smp.Unet('efficientnet-b4', in_channels=inchannels, classes=numclasses, activation='softmax')
    
    def forward(self, input):
        hidden_state = self.model.encoder(input)
        decoder_output = self.model.decoder(*hidden_state)

        masks = self.model.segmentation_head(decoder_output)
        '''
        if self.model.classification_head is not None:
            labels = self.model.classification_head(hidden_state[-1])
            return hidden_state, masks, labels
        '''
        
        return hidden_state, masks
       


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
    #def __init__(self, batch_size, transform, model1, model2):
    def __init__(self, batch_size, transform, model1):
        super(LitModelEfficientNet, self).__init__()
        self.batch_size = batch_size
        self.transform = transform

        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.MSELoss()

        #self.cnnexpertRGB = model1
        self.cnnexpertRGB = UNetExpert1(inchannels=3, numclasses=3)
        #self.cnnexpertDepth = model2
        #self.cnnexpertRGB = CNNExpert1(3, 3) ### model_weights[key.replace("auto_encoder.", "")] = model_weights.pop(key)
        self.cnnexpertDepth = CNNExpert1(1, 3) 
        self.cnnexpertLidar = CNNExpert1(2, 3)
        self.cnnexpertThermo = CNNExpert1(1, 3)
        #hidden_concat_dim = self.cnnexpertRGB.hidden_dim + self.cnnexpertDepth.hidden_dim + self.cnnexpertLidar.hidden_dim + self.cnnexpertThermo.hidden_dim
        hidden_concat_dim = self.cnnexpertDepth.hidden_dim + self.cnnexpertLidar.hidden_dim + self.cnnexpertThermo.hidden_dim ## definir este numero con base en el encoder

        self.gatingnetwork = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_concat_dim, 64)), #output 64 
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(64, 2)),   #output 2. Cambiar el 4 al tamanio
            ('softmax', nn.Softmax(dim=1))  ################### probabilidad. Que no rebase 1
        ]))

    def forward(self, x1, x2, x3, x4):
    #def forward(self, hidden_statergb, outrgb, hidden_statedepth, outdepth, x3, x4): ### here1 ## 1 input (x) con los dict

        hidden_statergb, outrgb = self.cnnexpertRGB(x1)
        hidden_statedepth, outdepth = self.cnnexpertDepth(x2)
        hidden_stateLidar, outLidar = self.cnnexpertLidar(x3)
        hidden_stateThermo, outThermo = self.cnnexpertThermo(x4)

        hidden_statergb = torch.cat(hidden_statergb, 0).reshape(len(hidden_statergb), 8, 3, 128, 128)

        #hidden_statergb = torch.as_tensor(hidden_statergb)



        ## coomo es el tamanio del encoder
        outconcatgating = torch.cat([hidden_statergb, hidden_statedepth, hidden_stateLidar, hidden_stateThermo], dim = -1)
        outconcatexpertclassifier = torch.stack([outrgb, outdepth, outLidar, outThermo], dim = -1) #(matrix de wxh) #stack las imaagenes
        gating = self.gatingnetwork(outconcatgating)
        gating = gating.unsqueeze(1)
        outfinal = outconcatexpertclassifier * gating # multiplicacion de matriz completa x array

        # 2 sopas
        # 1. regresar el que tenga maas prob
        # 2. el combinado



        ### en vez de sumar, agarrar el max(max probability, o max energy)
        return outfinal.sum(-1) ######## Regression instead of  classif. Check dim


    def train_dataloader(self):
        #trainset = torchvision.datasets.Kitti(root='./data', train=True, download=True, transform=self.transform)
        #trainset = ThermalDataset("~/Downloads/00")

        
        
        # trainset = MultiModalDataset(csv_file=dir_path + '/' + 'multimodal.csv', root_dir=dir_path + '/'+'multimodalfolder', transform=self.transform)
        #trainset = MultiModalDataset(csv_file=dir_path + '/' + 'multimodal.csv', root_dir=dir_path + '/'+'multimodalfolder', transform=self.transform)

        print("dir_path ", dir_path)
        print(dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00/fl_rgb/')
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00'

        trainset = MultiModalDataset(rgb_path= dir_path2 + '/' + 'fl_rgb', 
                                        thermo_path = dir_path2 + '/' + 'fl_ir_aligned',
                                        label_path= dir_path2 + '/' + 'fl_rgb_labels',
                                        transform= self.transform)       

        #print("dir_path ", dir_path)
        #trainset = MultiModalDataset(csv_file='multimodal.csv', root_dir='multimodalfolder', transform=self.transform)
        



        #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        #                                   download=True, transform=self.transform)
        



        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
        #                                    shuffle=True, num_workers=10, collate_fn = collate_multimodal) #2

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
    '''
    def training_step_before(self, train_batch, batch_idx): #### CUDA
        inputs, labels = train_batch
        #optimizer.zero_grad()
        batch_size = 8
        w, h = 128, 128
        #w=1392
        #h = 512
        #labels = torch.randint(0, 2, [batch_size]).cuda() #high exclusive low inclusive ###### aquii posiciones con funcioon GPS -> posvector
        #inputrgb = torch.rand(batch_size, 3, w, h).cuda()
        #inputdepth = torch.rand(batch_size, 1, w, h).cuda()
        #inputLidar = torch.rand(batch_size, 2, w, h).cuda()
        #inputThermo = torch.rand(batch_size, 1, w, h).cuda()

        labels = torch.rand(batch_size, 3)
        inputrgb = torch.rand(batch_size, 3, w, h)
        inputdepth = torch.rand(batch_size, 1, w, h)
        inputLidar = torch.rand(batch_size, 2, w, h)
        inputThermo = torch.rand(batch_size, 1, w, h)
        outputs = self(inputrgb, inputdepth, inputLidar, inputThermo) #forward(x1, x2, x3, x4)
        loss = self.criterion(outputs, labels)
        return loss
    '''
    
    def training_step(self, train_batch, batch_idx):
        #inputrgb, inputdepth, labels = train_batch
        inputrgb, labels = train_batch
       
        #inputs = train_batch
        #optimizer.zero_grad()
        batch_size = 8
        w, h = 128, 128
        #w=1392
        #h = 512

        #labels = torch.rand(batch_size, 3)
        #inputrgb = torch.rand(batch_size, 3, w, h)
        inputdepth = torch.rand(batch_size, 1, w, h)
        inputLidar = torch.rand(batch_size, 2, w, h)
        inputThermo = torch.rand(batch_size, 1, w, h)


        #print("lidar shape: ", inputLidar.shape)
        #print("rgb shape: ", inputrgb.shape)
        #print("depth shape: ", inputdepth.shape)
        #print(type(inputLidar))
        #print(type(inputrgb))

        ### add 1 channel to depth
        #inputdepth = np.expand_dims(inputdepth, axis=1)
        #print("depth shape2: ", inputdepth.shape)

        ## reorder rgb shape 111, 512, 1392, 3 to have 111, 3, 512, 1392
        #inputrgb = inputrgb.permute(0, 3, 1, 2)
        #print("rgb shape2: ", inputrgb.shape)

        
        #inputrgb = torch.rand(batch_size, 3, w, h)

        #outputs = self(inputs['rgb'], inputs['depth'], inputLidar, inputThermo) #forward(x1, x2, x3, x4)
        #loss = self.criterion(outputs, inputs['label'])

        outputs = self(inputrgb, inputdepth, inputLidar, inputThermo) #forward(x1, x2, x3, x4)
        #outputs = self(hidden_statergb, outrgb, hidden_statedepth, outdepth, inputLidar, inputThermo) #forward(x1, x2, x3, x4)
        loss = self.criterion(outputs, labels)
        return loss

        
        #histate, outputs = self(inputrgb) #forward(x1, x2, x3, x4)
        #loss = self.criterion(outputs, labels)
        #return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        outputs = self(images)
    
        _, predicted = torch.max(outputs.data, 1)

        loss = F.cross_entropy(predicted, labels)
        #loss = F.mse_loss(predicted, labels)

        return loss

##Revisar
#torch.save(model.state_dict(), save_path)