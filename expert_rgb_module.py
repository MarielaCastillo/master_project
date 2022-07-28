import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from skimage import io
from torch.utils.data import Dataset
import torchmetrics

dir_path = os.path.dirname(os.path.realpath(__file__))

class MultiModalDataset2(Dataset):
    def __init__(self, txt_file, file_path, label_path, transform_rgb=None, transform_thermo=None):
        self.annotations = pd.read_csv(txt_file, sep=" ", header=None)
        self.labels_path = label_path
        self.files_path = file_path
        self.transform_thermo = transform_thermo
        self.transform_rgb = transform_rgb

    def __getitem__(self, index):
        thermo_path = os.path.join(self.files_path, self.annotations.iloc[index, 0]+".jpeg")
        img_thermo = io.imread(thermo_path)

        file_name = self.annotations.iloc[index, 0].replace("_PreviewData", "")

        rgb_path = os.path.join(self.files_path, file_name+"_RGB.jpg")
        img_rgb = io.imread(rgb_path)

        #lbl_path = os.path.join(self.labels_path, file_name+"_labels.jpg")
        #label = io.imread(lbl_path)

        ndarray_from_file=np.load(self.labels_path+"/"+file_name+".npy")
        #label = torch.from_numpy(ndarray_from_file)
        #print(torch.unique(label))


        if self.transform_rgb:
            img_rgb = self.transform_rgb(img_rgb)
        if self.transform_thermo:
            img_thermo = self.transform_thermo(img_thermo)
            img_thermo = img_thermo[0].unsqueeze(dim=0)
        

        return img_rgb, img_thermo, ndarray_from_file, file_name
        # return img_rgb, img_thermo, label

    def __len__(self):
        return len(self.annotations)


class MultiModalDataset(Dataset):
    def __init__(self, rgb_path, thermo_path, label_path, transform_rgb=None, transform_thermo=None):
        self.rgb_path = rgb_path
        self.thermo_path = thermo_path
        self.label_path = label_path
        self.rgb_filepaths = [f for f in sorted(os.listdir(rgb_path))]
        self.thermo_filepaths = [f for f in sorted(os.listdir(thermo_path))]
        self.label_filepaths = [f for f in sorted(os.listdir(label_path))]
        self.transform_thermo= transform_thermo
        self.transform_rgb = transform_rgb

    def __getitem__(self, idx):
        img_rgb = io.imread(self.rgb_path+'/'+self.rgb_filepaths[idx])
        img_thermo = io.imread(self.thermo_path+'/'+self.thermo_filepaths[idx])
        img_thermo = np.expand_dims(img_thermo, axis=0).astype(np.int16)
        label = io.imread(self.label_path+'/'+self.label_filepaths[idx])
        if self.transform_rgb:
            img_rgb = self.transform_rgb(img_rgb)
        if self.transform_thermo:
            img_thermo = self.transform_thermo(img_thermo)

        return img_rgb, img_thermo, label

    def __len__(self):
        return len(self.rgb_filepaths)


class UNetExpert1(nn.Module):
    def __init__(self, inchannels, numclasses):
        super().__init__()
        self.inchannels = inchannels
        self.model = smp.Unet('efficientnet-b4', in_channels=self.inchannels, classes=numclasses)

    def forward(self, input):
        hidden_state = self.model.encoder(input)
        decoder_output = self.model.decoder(*hidden_state)

        masks = self.model.segmentation_head(decoder_output)

        return hidden_state, masks


# ###Model
class LitModelEfficientNetRgb(pl.LightningModule):
    def __init__(self, batch_size, transform, lr = 0.001):
        super(LitModelEfficientNetRgb, self).__init__()
        self.batch_size = batch_size
        self.transform_rgb = transform
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = lr

        # self.accuracy = torchmetrics.Accuracy()

        self.cnnexpert = UNetExpert1(inchannels=3, numclasses=5)

    def forward(self, x1): ### here1
        hiddenrgb, outrgb,  = self.cnnexpert(x1)
        return outrgb

    def train_dataloader(self):
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00'
        dir_path3 = dir_path + '/' + 'align'
        '''
        trainset = MultiModalDataset(rgb_path= dir_path2 + '/' + 'fl_rgb', 
                                        thermo_path = dir_path2 + '/' + 'fl_ir_aligned',
                                        label_path= dir_path2 + '/' + 'fl_rgb_labels',
                                        transform_rgb= self.transform_rgb)  
        '''

        trainset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_train.txt',
                                     file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     #label_path=dir_path + '/' + 'labels_ss',
                                     label_path=dir_path + '/' + 'labels_npy',
                                     transform_rgb=self.transform_rgb)  
        #'''       

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=4)                                                                             
        return trainloader

    def test_dataloader(self):
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/test/seq_01_day/00'
        dir_path3 = dir_path + '/' + 'align'

        '''
        testset = MultiModalDataset(rgb_path= dir_path2 + '/' + 'fl_rgb', 
                                        thermo_path = dir_path2 + '/' + 'fl_ir_aligned',
                                        label_path= dir_path2 + '/' + 'fl_rgb_labels',
                                        transform_rgb= self.transform_rgb)  
        '''
        
        testset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_validation.txt',
                                     file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     #label_path=dir_path + '/' + 'labels_ss',
                                     label_path=dir_path + '/' + 'labels_npy_val',
                                     transform_rgb=self.transform_rgb)         

        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=4)   
        return testloader

    ### optimizers and schedulers
    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer

    def training_step(self, train_batch):
        viz_pred = False
        input_rgb, _, labels, file_name = train_batch
        #optimizer.zero_grad()

        outputs = self(input_rgb)
        if viz_pred:
            pred = outputs.argmax(axis=1).detach().cpu().numpy()
            pred = pred * 255 / pred.max()
            plt.imshow(pred[0])
            plt.show()

        #print(torch.unique(labels))

        loss = self.criterion(outputs, labels.long())

        logs={"train_loss": loss}

        # log step metric
        #self.accuracy(outputs, labels)
        #self.log('train_acc_step', self.accuracy)
        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html

        #values = {"loss": loss, "acc": acc, "metric_n": metric_n}  # add more items if needed
        #values = {"loss": loss}  # add more items if needed
        #self.log_dict(values)

        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #     tensorboard_logs = {'loss': avg_loss,"Accuracy": correct/total}  # to paste in page
        # https://www.pytorchlightning.ai/blog/tensorboard-with-pytorch-lightning
        # tensorboard_logs = {'loss': logs,"Accuracy": correct/total}

        return loss

    # def training_epoch_end(self, outs):
        # log epoch metric
        # self.log('train_acc_epoch', self.accuracy)

    def test_step(self, test_batch, batch_idx):
        viz_pred = True
        images, _, labels, file_name = test_batch
        outputs = self(images)

        if viz_pred:
            # Labels
            lbl = labels.detach().cpu().numpy()  # detach es para graficar y transformar a numpy
            # plt.imshow(lbl[0])
            #plt.show()

            # Prediction
            pred = outputs.argmax(axis=1).detach().cpu().numpy()
            if pred.max() != 0:
                pred = pred * 255 / pred.max()
            else:
                pred = pred * 0

            plt.imsave("eval_rgb/"+file_name[0]+"_eval_label.png", lbl[0])
            plt.imsave("eval_rgb/"+file_name[0]+"_eval_pred_rgb.png", pred[0])
            
            viz_pred = False
            # plt.imshow(pred[0])
            # plt.show()

        loss = self.criterion(outputs, labels.long())

        return loss