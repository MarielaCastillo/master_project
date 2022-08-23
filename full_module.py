import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torchmetrics import JaccardIndex
from torchmetrics import Recall
from torchmetrics import Precision
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead

# from expert_rgb_module import MultiModalDataset
from expert_rgb_module import MultiModalDataset2

from matplotlib.colors import ListedColormap

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LitModelEfficientNetFull(pl.LightningModule):
    def __init__(self, batch_size, transform_rgb, transform_thermo, model1, model2, checkpoint_epochs=""):
        super(LitModelEfficientNetFull, self).__init__()
        self.batch_size = batch_size
        self.transform_rgb = transform_rgb
        self.transform_thermo = transform_thermo
        self.checkpoint_epochs = checkpoint_epochs

        self.count_rgb = 0
        self.count_thermo = 0


        self.criterion = nn.CrossEntropyLoss()

        self.cnnexpertRGB = model1
        self.cnnexpertThermo = model2
        
        encoder_channels = list(self.cnnexpertRGB.model.encoder.out_channels)
        #encoder_channels = (3, 48, 32, 56, 160, 448)
        encoder_channels = [2 * a for a in encoder_channels]
        decoder_channels = (256, 128, 64, 32, 16)
        self.decoder = UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                n_blocks=5,
                use_batchnorm=True,
                center=False,
                attention_type=None)

        self.head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=2,
                kernel_size=3,
                activation='softmax'
            )

    def forward(self, x1, x2=None):
        hidden_states_rgb, outrgb = self.cnnexpertRGB(x1)
        hidden_states_depth, outdepth = self.cnnexpertThermo(x2)

        input_gating = []

        for hidden_state_rgb, hidden_state_depth in zip(hidden_states_rgb, hidden_states_depth):
            input_gating.append(torch.cat([hidden_state_rgb, hidden_state_depth], dim=1))
        gating = self.decoder(*input_gating)
        gating = self.head(gating)
        outfinal = gating[:, 0] * outrgb + gating[:, 1] * outdepth

        boolean_tensor = gating[:,0] < gating[:,1]
        gating_rgb_count = torch.numel(gating[:,0])  # 512*640 = 327680
        # gating_thermo_count = torch.numel(gating[:,1])  # 512*640 = 327680
        # total_elements = gating_rgb_count + gating_thermo_count  # 655360 (512*640 + 512*640)

        total_elements = gating_rgb_count  # could be rgb or thermo, since they have the same size. 

        is_rgb_better = gating[:,0] > gating[:,1]
        rgb_better = torch.count_nonzero(is_rgb_better)

        is_thermo_better = gating[:,0] < gating[:,1]
        thermo_better = torch.count_nonzero(is_thermo_better)

        percentage_rgb = int(rgb_better)/total_elements * 100
        percentage_thermo= int(thermo_better)/total_elements * 100

        return outfinal, percentage_rgb, percentage_thermo

        # contador rgb, contador thermo y %
        # guardar un ejemplo donde ganoo rgb y otra de thermo
        # poner nombre de la imagen si se puede

    def train_dataloader(self):
        dir_path3 = dir_path + '/' + 'align'                        

        trainset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_train.txt',
                                     #file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     file_path=dir_path3 + '/' + 'JPEGImages',
                                     label_path=dir_path + '/' + 'labels_npy',
                                     transform_rgb=self.transform_rgb,
                                     transform_thermo=self.transform_thermo)  

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=0)  # 2
        return trainloader

    def test_dataloader(self):
        dir_path3 = dir_path + '/' + 'align'
        
        testset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_validation.txt',
                                     #file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     file_path=dir_path3 + '/' + 'JPEGImages',
                                     label_path=dir_path + '/' + 'labels_npy_val',
                                     transform_rgb=self.transform_rgb,
                                     transform_thermo=self.transform_thermo) 

        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=10)
        return testloader

    # ## optimizers and schedulers
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
    
    def training_step(self, train_batch):
        viz_pred=False
        input_rgb, input_thermo, labels, file_name = train_batch
        labels = labels.long()

        outputs, perc_rgb, perc_thermo = self(input_rgb, input_thermo)

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
            
            print("RGB input size", input_rgb.size(), input_rgb.max(), input_rgb.min())
            plt.imshow(input_rgb[0].permute(1, 2, 0))
            print("RGB")
            plt.show()

            print("Thermo input size", input_thermo.size(), input_thermo.max(), input_thermo.min())
            plt.imshow(input_thermo[0].permute(1, 2, 0))
            print("Thermo")
            plt.show()

            print("labels are ", labels.size(), labels.max(), labels.min())
            plt.imshow(labels[0], cmap=colourmap, vmin=imin, vmax=imax)
            print("Labels")
            plt.show()

            print("output is")
            
            pred = outputs.argmax(axis=1).detach().cpu().numpy()
            # pred = pred * 255 / pred.max()
            plt.imshow(pred[0], cmap=colourmap, vmin=imin, vmax=imax)
            plt.show()

        # print("perc_rgb", perc_rgb, "perc_thermo", perc_thermo)
        loss = self.criterion(outputs, labels)

        # IoU
        jaccard = JaccardIndex(num_classes=5).to(device)
        iou = jaccard(outputs, labels)

        # Recall = TP/(TP+FN)
        metric1 = Recall(num_classes=5, mdmc_average="samplewise")
        recall = metric1(outputs, labels)

        # Precision = TP/(TP+FP)
        metric2 = Precision(num_classes=5, mdmc_average="samplewise")
        precision = metric2(outputs, labels)

        # Accuracy = (TP + TN)/Total
        acc = accuracy(outputs, labels.long())

        if perc_rgb > perc_thermo:
            self.count_rgb = self.count_rgb + 1
        else:
            self.count_thermo = self.count_thermo + 1

        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        
        metrics = {"train_acc":acc, "train_loss":loss,  "iou":iou,
                                                        "recall": recall,
                                                        "precision":precision,
                                                        "count_rgb":self.count_rgb, 
                                                        "count_thermo":self.count_thermo}
        self.log_dict(metrics)

        return loss

    def test_step(self, test_batch, dataloader_idx):
        # check this
        viz_pred = True
        input_rgb, input_thermo, labels, file_name = test_batch
        labels = labels.long()

        outputs, perc_rgb, perc_thermo = self(input_rgb, input_thermo)
        loss = self.criterion(outputs, labels)

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
            # Labels
            lbl = labels.detach().cpu().numpy()  # detach es para graficar y transformar a numpy
            # plt.imshow(lbl[0])
            #plt.show()

            # Prediction
            pred = outputs.argmax(axis=1).detach().cpu().numpy()

            plt.imshow(lbl[0], cmap=colourmap, vmin=imin, vmax=imax)
            plt.show()
            plt.imshow(pred[0], cmap=colourmap, vmin=imin, vmax=imax)
            plt.show()

            # plt.imsave("eval_full/"+file_name[0]+"_eval_label.png", lbl[0])
            # plt.imsave("eval_full/"+file_name[0]+"_eval_pred_full.png", pred[0])
            
            plt.imsave("./img_eval_full/"+self.checkpoint_epochs+"_"+file_name[0]+"_eval_label.png", lbl[0], cmap=colourmap, vmin=imin, vmax=imax)
            plt.imsave("./img_eval_full/"+self.checkpoint_epochs+"_"+file_name[0]+"_eval_pred_full.png", pred[0], cmap=colourmap, vmin=imin, vmax=imax)
            
            viz_pred = False

        # _, predicted = torch.max(outputs.data, 1)
        # loss = F.cross_entropy(predicted, labels)

        acc = accuracy(outputs, labels.long())
        
        # IoU
        jaccard = JaccardIndex(num_classes=5).to(device)
        iou = jaccard(outputs, labels)

        # Recall = TP/(TP+FN)
        metric1 = Recall(num_classes=5, mdmc_average="samplewise")
        recall = metric1(outputs, labels)

        # Precision = TP/(TP+FP)
        metric2 = Precision(num_classes=5, mdmc_average="samplewise")
        precision = metric2(outputs, labels)

        # Accuracy = (TP + TN)/Total
        acc = accuracy(outputs, labels.long())

        if perc_rgb > perc_thermo:
            self.count_rgb = self.count_rgb + 1
        else:
            self.count_thermo = self.count_thermo + 1

        metrics = {"val_acc":acc, "val_loss":loss,  "iou":iou,
                                                        "recall": recall,
                                                        "precision":precision,
                                                        "count_rgb":self.count_rgb, 
                                                        "count_thermo":self.count_thermo}
        self.log_dict(metrics)


        return loss
