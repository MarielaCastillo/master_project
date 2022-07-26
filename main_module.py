import torch
import torch.nn as nn
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

        return outfinal

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

        trainset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_train.txt',
                                     file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     label_path=dir_path + '/' + 'labels_npy',
                                     transform_rgb=self.transform_rgb,
                                     transform_thermo=self.transform_thermo)  

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=10)  # 2
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

        testset = MultiModalDataset2(txt_file=dir_path3 + '/' + 'align_train.txt',
                                     file_path=dir_path3 + '/' + 'AnnotatedImages',
                                     label_path=dir_path + '/' + 'labels_npy',
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
        input_rgb, input_thermo, labels = train_batch
        labels = labels.long()

        outputs = self(input_rgb, input_thermo)
        loss = self.criterion(outputs, labels)
        return loss

    def test_step(self, test_batch):
        # check this
        input_rgb, input_thermo, labels = test_batch
        labels = labels.long()

        outputs = self(input_rgb, input_thermo)
        loss = self.criterion(outputs, labels)

        # _, predicted = torch.max(outputs.data, 1)
        # loss = F.cross_entropy(predicted, labels)

        return loss
