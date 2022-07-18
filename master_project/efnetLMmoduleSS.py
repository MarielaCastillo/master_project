from tabnanny import check
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from efnetLMmoduleExpert1SS import MultiModalDataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

####Model
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
                activation='softmax'
            )

    def forward(self, x1, x2=None, x3=None, x4=None):
        hidden_states_rgb, outrgb = self.cnnexpertRGB(x1)
        hidden_states_depth, outdepth = self.cnnexpertThermo(x2)

        input_gating = []
        for hidden_state_rgb, hidden_state_depth in zip(hidden_states_rgb, hidden_states_depth):
            input_gating.append(torch.cat([hidden_state_rgb, hidden_state_depth], dim = 1))
        gating = self.decoder(*input_gating)
        gating = self.head(gating)

        # 2 sopas
        # 1. regresar el que tenga maas prob

        # 2. el combinado
        outfinal = gating[:, 0] * outrgb + gating[:, 1] * outdepth

        return outfinal  # here breakpoint for visualizer

    def train_dataloader(self):
        print("dir_path ", dir_path)
        print(dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00/fl_rgb/')
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00'

        trainset = MultiModalDataset(rgb_path= dir_path2 + '/' + 'fl_rgb', 
                                        thermo_path = dir_path2 + '/' + 'fl_ir_aligned',
                                        label_path= dir_path2 + '/' + 'fl_rgb_labels',
                                        transform_rgb= self.transform_rgb, transform_thermo=self.transform_thermo)       

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=10) #2
        return trainloader

    def test_dataloader(self):
        print("dir_path ", dir_path)
        print(dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00/fl_rgb/')
        dir_path2 = dir_path + '/' + 'thermaldatasetfolder/train/seq_00_day/00'

        testset = MultiModalDataset(rgb_path= dir_path2 + '/' + 'fl_rgb', 
                                        thermo_path = dir_path2 + '/' + 'fl_ir_aligned',
                                        label_path= dir_path2 + '/' + 'fl_rgb_labels',
                                        transform_rgb= self.transform_rgb, transform_thermo=self.transform_thermo)       




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
        input_rgb, input_thermo, labels = train_batch
        labels = labels.long()

        outputs = self(input_rgb, input_thermo)  # forward(x1, x2, x3, x4) ### CAMBIAR
        loss = self.criterion(outputs, labels)
        return loss

    def test_step(self, test_batch, batch_idx):
        # check this
        images, labels = test_batch
        outputs = self(images)
    
        _, predicted = torch.max(outputs.data, 1)

        loss = F.cross_entropy(predicted, labels)

        return loss