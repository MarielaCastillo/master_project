import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics.functional import accuracy
from torchmetrics import JaccardIndex
from torchmetrics import Recall
from torchmetrics import Precision

from expert_rgb_module import UNetExpert1
from expert_rgb_module import MultiModalDataset
from expert_rgb_module import MultiModalDataset2

from matplotlib.colors import ListedColormap

import numpy as np

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScaleThermal:
    def __init__(self, max_value=30000) -> None:
        self.max_value = max_value
    
    def __call__(self,  input):
        return input / self.max_value


# ###Model
class LitModelEfficientNetThermo(pl.LightningModule):
    def __init__(self, batch_size, transform, checkpoint_epochs=""):
        super(LitModelEfficientNetThermo, self).__init__()
        self.batch_size = batch_size
        self.transform_thermo = transform
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_epochs = checkpoint_epochs

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
                                     #file_path=dir_path3 + '/' + 'AnnotatedImages',
                                      file_path=dir_path3 + '/' + 'JPEGImages',
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
                                     # file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     file_path=dir_path3 + '/' + 'JPEGImages',
                                     # label_path=dir_path + '/' + 'labels_ss',
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
        viz_pred=False
        _, input_thermo, labels, file_name = train_batch

        outputs = self(input_thermo)

        if viz_pred:
            '''
            print("THermo input size", input_thermo.size(), input_thermo.max(), input_thermo.min())
            plt.imshow(input_thermo[0].permute(1, 2, 0))
            plt.show()
            print("labels are ", labels.size(), labels.max(), labels.min())
            plt.imshow(input_thermo[0].permute(1, 2, 0))
            plt.show()
            print("output is")
            '''
            pred = outputs.argmax(axis=1).detach().cpu().numpy()
            pred = pred * 255 / pred.max()
            plt.imshow(pred[0])
            plt.show()

        loss = self.criterion(outputs, labels.long())

        # IoU
        jaccard = JaccardIndex(num_classes=5).to(device)
        iou = jaccard(outputs, labels.long())

        # Recall = TP/(TP+FN)
        metric1 = Recall(num_classes=5, mdmc_average="samplewise")
        recall = metric1(outputs, labels.long())

        # Precision = TP/(TP+FP)
        metric2 = Precision(num_classes=5, mdmc_average="samplewise")
        precision = metric2(outputs, labels.long())

        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        acc = accuracy(outputs, labels.long())
        metrics = {"train_acc": acc, "train_loss": loss, "iou":iou,
                                                        "recall": recall,
                                                        "precision":precision
                                                        }
        self.log_dict(metrics)

        return loss

    def test_step(self, test_batch, batch_idx):
        viz_pred = True
        _, images, labels, file_name = test_batch
        outputs = self(images)
    
        # _, predicted = torch.max(outputs.data, 1)

        if viz_pred:
            label_dict = {0: 'unlabelled',
                        1: 'car',
                        2: 'person',
                        3: 'bycicle',
                        4: 'dog'}
            color_dict = {0: 'black',
                        1: 'blue',
                        2: 'yellow',
                        3: 'lime',
                        4: 'red'}

            imin = min(label_dict)
            imax = max(label_dict)

            colourmap = ListedColormap(color_dict.values())

            lbl = labels.detach().cpu().numpy()

            pred = outputs.argmax(axis=1).detach().cpu().numpy()
            '''
            # values, counts = np.unique(lbl, return_counts=True)
            # values2, counts2 = np.unique(pred, return_counts=True)

            # print(file_name)
            # print("values", values, "counts", counts)
            # print("values2", values2, "counts2", counts2)
            # print()
            '''

            # plt.imshow(lbl[0], cmap=colourmap, vmin=imin, vmax=imax)
            # plt.show()
            # plt.imshow(pred[0], cmap=colourmap, vmin=imin, vmax=imax)
            # plt.show()

            plt.imsave("./img_eval_thermo/"+self.checkpoint_epochs+"_"+file_name[0]+"_eval_label.png", lbl[0], cmap=colourmap, vmin=imin, vmax=imax)
            plt.imsave("./img_eval_thermo/"+self.checkpoint_epochs+"_"+file_name[0]+"_eval_pred_thermo.png", pred[0], cmap=colourmap, vmin=imin, vmax=imax)
            
            viz_pred = False

        loss = self.criterion(outputs, labels.long())
        

        # loss = F.mse_loss(predicted, labels)


        # IoU
        jaccard = JaccardIndex(num_classes=5).to(outputs.device)
        iou = jaccard(outputs, labels.long())

        # Recall = TP/(TP+FN)
        metric1 = Recall(num_classes=5, mdmc_average="samplewise")
        recall = metric1(outputs, labels.long())

        # Precision = TP/(TP+FP)
        metric2 = Precision(num_classes=5, mdmc_average="samplewise")
        precision = metric2(outputs, labels.long())

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        acc = accuracy(outputs, labels.long())
        metrics = {"val_acc": acc, "val_loss": loss, "val_iou":iou,
                                                        "val_recall": recall,
                                                        "val_precision":precision
                                                        }
        self.log_dict(metrics)


        return loss
