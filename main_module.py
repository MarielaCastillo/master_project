import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
# from expert_rgb_module import MultiModalDataset
from expert_rgb_module import MultiModalDataset2

from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


class LitModelEfficientNetFull(pl.LightningModule):
    def __init__(self, batch_size, transform_rgb, transform_thermo, model1, model2):
        super(LitModelEfficientNetFull, self).__init__()
        self.batch_size = batch_size
        self.transform_rgb = transform_rgb
        self.transform_thermo = transform_thermo

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
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00'
        dir_path3 = dir_path + '/' + 'align'

        '''
        trainset = MultiModalDataset(rgb_path=dir_path2 + '/' + 'fl_rgb',
                                     thermo_path=dir_path2 + '/' + 'fl_ir_aligned',
                                     label_path=dir_path2 + '/' + 'fl_rgb_labels',
                                     transform_rgb=self.transform_rgb,
                                     transform_thermo=self.transform_thermo)
        '''                             

        trainset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_validation.txt',
                                     file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     label_path=dir_path + '/' + 'labels_npy_val',
                                     transform_rgb=self.transform_rgb,
                                     transform_thermo=self.transform_thermo)  

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=0)  # 2
        return trainloader

    def test_dataloader(self):
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00'
        dir_path3 = dir_path + '/' + 'align'
        
        '''
        testset = MultiModalDataset(rgb_path=dir_path2 + '/' + 'fl_rgb',
                                    thermo_path=dir_path2 + '/' + 'fl_ir_aligned',
                                    label_path=dir_path2 + '/' + 'fl_rgb_labels',
                                    transform_rgb=self.transform_rgb,
                                    transform_thermo=self.transform_thermo)
        '''

        testset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_validation.txt',
                                     file_path=dir_path3 + '/' + 'AnnotatedImages',
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
        input_rgb, input_thermo, labels, file_name = train_batch
        labels = labels.long()

        outputs, perc_rgb, perc_thermo = self(input_rgb, input_thermo)
        print("perc_rgb", perc_rgb, "perc_thermo", perc_thermo)
        loss = self.criterion(outputs, labels)
        return loss

    def test_step(self, test_batch, dataloader_idx):
        # check this
        viz_pred = True
        input_rgb, input_thermo, labels, file_name = test_batch
        labels = labels.long()

        outputs = self(input_rgb, input_thermo)
        loss = self.criterion(outputs, labels)

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

            # plt.imsave("eval_full/"+file_name[0]+"_eval_label.png", lbl[0])
            # plt.imsave("eval_full/"+file_name[0]+"_eval_pred_full.png", pred[0])
            
            plt.imsave(file_name[0]+"_eval_label.png", lbl[0])
            plt.imsave(file_name[0]+"_eval_pred_full.png", pred[0])
            
            viz_pred = False

        # _, predicted = torch.max(outputs.data, 1)
        # loss = F.cross_entropy(predicted, labels)

        return loss
