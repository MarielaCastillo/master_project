import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from expert_rgb_module import UNetExpert1

from expert_rgb_module import MultiModalDataset

import matplotlib.pyplot as plt

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class ScaleThermal():
    def __init__(self, max_value=30000) -> None:
        self.max_value = max_value
    
    def __call__(self,  input):
        return input / self.max_value
   
####Model
class LitModelEfficientNetThermo(pl.LightningModule):
    def __init__(self, batch_size, transform):
        super(LitModelEfficientNetThermo, self).__init__()
        self.batch_size = batch_size
        self.transform_thermo = transform
        self.criterion = nn.CrossEntropyLoss()

        self.cnnexpert = UNetExpert1(inchannels=1, numclasses=13)

    def forward(self, x1): ### here1
        _, output = self.cnnexpert(x1) #why
        return output

    def train_dataloader(self):
        print("dir_path ", dir_path)
        print(dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00/fl_rgb/')
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00'

        trainset = MultiModalDataset(rgb_path= dir_path2 + '/' + 'fl_rgb', 
                                        thermo_path = dir_path2 + '/' + 'fl_ir_aligned',
                                        label_path= dir_path2 + '/' + 'fl_rgb_labels',
                                        transform_thermo= self.transform_thermo)  

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=3)                                                                             
        return trainloader

    def test_dataloader(self):
        print("dir_path ", dir_path)
        print(dir_path + '/' + 'thermaldatasetfolder/test/seq_01_day/00/fl_rgb/')
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/test/seq_01_day/00'

        testset = MultiModalDataset(rgb_path= dir_path2 + '/' + 'fl_rgb', 
                                        thermo_path = dir_path2 + '/' + 'fl_ir_aligned',
                                        label_path= dir_path2 + '/' + 'fl_rgb_labels',
                                        transform_thermo= self.transform_thermo)  
                                        
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=3)   
        return testloader

    ### optimizers and schedulers
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        viz_pred = False
        _, input_thermo, labels = train_batch
        #optimizer.zero_grad()

        outputs = self(input_thermo) #forward(x1, x2, x3, x4)
        if viz_pred:
            pred = outputs.argmax(axis=1).detach().cpu().numpy()
            pred = pred * 255 / pred.max()
            plt.imshow(pred[0])
            plt.show()

        loss = self.criterion(outputs, labels.long())
        return loss

    def test_step(self, test_batch, batch_idx):
        viz_pred = True
        _, images, labels = test_batch
        outputs = self(images)
    
        #_, predicted = torch.max(outputs.data, 1)

        if viz_pred:
            lbl = labels.detach().cpu().numpy() # detach es para graficar y transformar a numpy
            #lbl = lbl * 255 / lbl.max()
            plt.imshow(lbl[0])
            plt.show()

            pred = outputs.argmax(axis=1).detach().cpu().numpy() # detach es para graficar y transformar a numpy
            pred = pred * 255 / pred.max()
            plt.imshow(pred[0])
            plt.show()

        loss = self.criterion(outputs, labels.long())
        #loss = F.mse_loss(predicted, labels)

        return loss
