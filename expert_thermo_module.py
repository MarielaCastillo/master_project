import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
from expert_rgb_module import UNetExpert1

from expert_rgb_module import MultiModalDataset
from expert_rgb_module import MultiModalDataset2

import matplotlib.pyplot as plt

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


class ScaleThermal:
    def __init__(self, max_value=30000) -> None:
        self.max_value = max_value
    
    def __call__(self,  input):
        return input / self.max_value


# ###Model
class LitModelEfficientNetThermo(pl.LightningModule):
    def __init__(self, batch_size, transform):
        super(LitModelEfficientNetThermo, self).__init__()
        self.batch_size = batch_size
        self.transform_thermo = transform
        self.criterion = nn.CrossEntropyLoss()

        self.cnnexpert = UNetExpert1(inchannels=1, numclasses=5)

    def forward(self, x1):
        _, output = self.cnnexpert(x1)
        return output

    def train_dataloader(self):
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00'
        dir_path3 = dir_path + '/' + 'align'
        
        '''
        trainset = MultiModalDataset(rgb_path=dir_path2 + '/' + 'fl_rgb',
                                     thermo_path=dir_path2 + '/' + 'fl_ir_aligned',
                                     label_path=dir_path2 + '/' + 'fl_rgb_labels',
                                     transform_thermo=self.transform_thermo)

        '''
        trainset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_train.txt',
                                     file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     #label_path=dir_path + '/' + 'labels_ss',
                                     label_path=dir_path + '/' + 'labels_npy',
                                     transform_thermo=self.transform_thermo)  
        

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True, num_workers=3)
        return trainloader

    def test_dataloader(self):
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/test/seq_01_day/00'
        dir_path3 = dir_path + '/' + 'align'

        '''
        testset = MultiModalDataset(rgb_path=dir_path2 + '/' + 'fl_rgb',
                                    thermo_path=dir_path2 + '/' + 'fl_ir_aligned',
                                    label_path=dir_path2 + '/' + 'fl_rgb_labels',
                                    transform_thermo=self.transform_thermo)
        '''

        testset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_validation.txt',
                                     file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     #label_path=dir_path + '/' + 'labels_ss',
                                     label_path=dir_path + '/' + 'labels_npy_val',
                                     transform_thermo=self.transform_thermo)         
                                        
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False, num_workers=3)
        return testloader

    # ## optimizers and schedulers
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        viz_pred = False
        _, input_thermo, labels = train_batch

        outputs = self(input_thermo)
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
    
        # _, predicted = torch.max(outputs.data, 1)

        if viz_pred:
            lbl = labels.detach().cpu().numpy()
            plt.imshow(lbl[0])
            plt.show()

            pred = outputs.argmax(axis=1).detach().cpu().numpy()
            if pred.max() != 0:
                pred = pred * 255 / pred.max()
            else:
                pred = 0

            plt.imsave("eval_rgb.png", pred[0])
            # plt.imsave("eval_thermo_", pred[0], format='png')
            # plt.imshow(pred[0])
            # plt.show()

        loss = self.criterion(outputs, labels.long())
        # loss = F.mse_loss(predicted, labels)

        return loss
