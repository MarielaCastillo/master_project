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
from efnetLMmoduleExpert1SS import MultiModalDataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead

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

####Model
class LitModelEfficientNetFull(pl.LightningModule):
    #https://github.com/gthparch/edgeBench/blob/master/pytorch/models/cifarnet.py#L23
    #def __init__(self, batch_size, transform, model1, model2):
    def __init__(self, batch_size, transform, model1, model2):
        super(LitModelEfficientNetFull, self).__init__()
        self.batch_size = batch_size
        self.transform = transform

        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.MSELoss()

        self.cnnexpertRGB = model1
        #self.cnnexpertRGB = UNetExpert1(inchannels=3, numclasses=3)
        #self.cnnexpertDepth = model2
        #self.cnnexpertRGB = CNNExpert1(3, 3) ### model_weights[key.replace("auto_encoder.", "")] = model_weights.pop(key)
        self.cnnexpertDepth = model2
        #self.cnnexpertLidar = CNNExpert1(2, 3)
        #self.cnnexpertThermo = CNNExpert1(1, 3)
        #hidden_concat_dim = self.cnnexpertRGB.hidden_dim + self.cnnexpertDepth.hidden_dim + self.cnnexpertLidar.hidden_dim + self.cnnexpertThermo.hidden_dim
        
        #hidden_concat_dim = self.cnnexpertDepth.hidden_dim + self.cnnexpertLidar.hidden_dim + self.cnnexpertThermo.hidden_dim ## definir este numero con base en el encoder

        encoder_channels = list(self.cnnexpertRGB.model.encoder.out_channels)
        encoder_channels = [ 2 * a  for a in encoder_channels]
        decoder_channels = (256, 128, 64, 32, 16)
        self.decoder = UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                n_blocks=5,
                use_batchnorm= True,
                center= False,
                attention_type= None)

        self.head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=2,
                kernel_size=3,
            )

    def forward(self, x1, x2=None, x3=None, x4=None):
    #def forward(self, hidden_statergb, outrgb, hidden_statedepth, outdepth, x3, x4): ### here1 ## 1 input (x) con los dict

        hidden_states_rgb, outrgb = self.cnnexpertRGB(x1)
        hidden_states_depth, outdepth = self.cnnexpertRGB(x2) #### cambiar a x2
        #hidden_stateLidar, outLidar = self.cnnexpertLidar(x3)
        #hidden_stateThermo, outThermo = self.cnnexpertThermo(x4)

        #hidden_statergb = torch.as_tensor(hidden_statergb)



        ## coomo es el tamanio del encoder
        # outconcatexpertclassifier = torch.stack([outrgb, outdepth], dim = -1) #(matrix de wxh) #stack las imaagenes
        input_gating = []
        for hidden_state_rgb, hidden_state_depth in zip(hidden_states_rgb, hidden_states_depth):
            input_gating.append(torch.cat([hidden_state_rgb, hidden_state_depth], dim = 1))
        gating = self.decoder(*input_gating)
        gating = self.head(gating)

        # 2 sopas
        # 1. regresar el que tenga maas prob

        # 2. el combinado
        outfinal = gating[:, 0] * outrgb + gating[:, 1] * outdepth



        ### en vez de sumar, agarrar el max(max probability, o max energy)
        return outfinal ######## Regression instead of  classif. Check dim


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
        labels = labels.long()
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

        outputs = self(inputrgb, inputrgb) #forward(x1, x2, x3, x4)
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