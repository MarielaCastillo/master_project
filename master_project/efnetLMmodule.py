import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from collections import OrderedDict

import pandas as pd
import os
from skimage import io

import numpy as np

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None): ### leer los nombres
        self.annotations = pd.read_csv(csv_file, dtype=str)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index): ##collate #should it be reading one at a time???
        img_path = os.path.join(self.root_dir, f'{self.annotations.iloc[index, 1]}_rgb.png')
        imagergb = io.imread(img_path)

        img_path = os.path.join(self.root_dir, f'{self.annotations.iloc[index, 0]}.png')
        imagedepth = io.imread(img_path)
        
        #img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        #lidar = pending


        y_label = torch.tensor(int(self.annotations.iloc[index, 4]))

        if self.transform:
            imagergb = self.transform(imagergb)
            imagedepth = self.transform(imagedepth)


        #return [imagergb, y_label]

        return [imagergb, imagedepth, y_label]


'''
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return rgb, bw, lidar, target

    def __len__(self) -> int:
        return len(self.data)
    
'''



class CNNExpert1(nn.Module):
    def __init__(self, inchannel, numclasses):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b4', in_channels=inchannel)
        self.disable_gradients(self.efficient_net)
        self.e2h = nn.Linear(1792, 64)
        self.hidden_dim = 64
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=256),
            #nn.Linear(in_features=1792, out_features=256),
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
        return hidden_state, self.fc(hidden_state) #8x2 1792x28 error

class CNNExpert2(nn.Module):
    def __init__(self, numclasses):
        #input channel should always be 3 for now
        super(CNNExpert2, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.hidden_dim = 1000
        self.despuespool = nn.Sequential(OrderedDict([

            ('dropout4', nn.Dropout()),
            ('relu4', nn.ReLU(True)),
            ('fc5', nn.Linear(1000, numclasses)) #1000 porque resnet18 output is 1000
        ]))

    def forward(self, input):
        hidden_state = self.model(input) 
        #return hidden_state, self.despuespool(hidden_state) #32x53824  and 1600x384
        return hidden_state, self.despuespool(hidden_state) #8x1000 and 64x2

class CNNExpert3(nn.Module):
    def __init__(self, inchannel, numclasses):
        super(CNNExpert3, self).__init__()
        self.antespool = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inchannel, 64, kernel_size=5)),
            ('batch1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=5)),
            ('batch2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3', nn.Conv2d(64, 32, kernel_size=5)),
            ('batch3', nn.BatchNorm2d(32)),
            ('relu3', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', nn.Flatten()),
            ('fc3', nn.Linear(4608, 384)), #53824, 9216
            ('dropout3', nn.Dropout()),
            ('relu3', nn.ReLU(True)),
            ('fc4', nn.Linear(384, 64)),
        ]))
        self.hidden_dim = 64
        self.despuespool = nn.Sequential(OrderedDict([

            ('dropout4', nn.Dropout()),
            ('relu4', nn.ReLU(True)),
            ('fc5', nn.Linear(64, numclasses))
        ]))

    def forward(self, input):
        hidden_state = self.antespool(input)
        return hidden_state, self.despuespool(hidden_state) #32x53824  and 1600x384

class CNNExpert4(nn.Module):
    def __init__(self, inchannel, numclasses):
        super(CNNExpert4, self).__init__()
        self.antespool = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inchannel, 64, kernel_size=5)),
            ('batch1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=5)),
            ('batch2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3', nn.Conv2d(64, 32, kernel_size=5)),
            ('batch3', nn.BatchNorm2d(32)),
            ('relu3', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', nn.Flatten()),
            ('fc3', nn.Linear(4608, 384)), #53824, 9216
            ('dropout3', nn.Dropout()),
            ('relu3', nn.ReLU(True)),
            ('fc4', nn.Linear(384, 64)),
        ]))
        self.hidden_dim = 64
        self.despuespool = nn.Sequential(OrderedDict([

            ('dropout4', nn.Dropout()),
            ('relu4', nn.ReLU(True)),
            ('fc5', nn.Linear(64, numclasses)) 
        ]))

    def forward(self, input):
        hidden_state = self.antespool(input)
        return hidden_state, self.despuespool(hidden_state) #32x53824  and 1600x384
        
####Model
class LitModelEfficientNet(pl.LightningModule):
    #https://github.com/gthparch/edgeBench/blob/master/pytorch/models/cifarnet.py#L23
    def __init__(self, batch_size, transform):
        super(LitModelEfficientNet, self).__init__()
        self.batch_size = batch_size
        self.transform = transform
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()

        self.cnnexpertRGB = CNNExpert1(3, 3) ### change to 1
        self.cnnexpertDepth = CNNExpert1(1, 3) 
        self.cnnexpertLidar = CNNExpert1(2, 3)
        self.cnnexpertThermo = CNNExpert1(1, 3)
        hidden_concat_dim = self.cnnexpertRGB.hidden_dim + self.cnnexpertDepth.hidden_dim + self.cnnexpertLidar.hidden_dim + self.cnnexpertThermo.hidden_dim

        self.gatingnetwork = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_concat_dim, 64)), #output 64
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(64, 4)),   #output 2
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, x1, x2, x3, x4):
        hidden_statergb, outrgb = self.cnnexpertRGB(x1)
        hidden_statedepth, outdepth = self.cnnexpertDepth(x2)
        hidden_stateLidar, outLidar = self.cnnexpertLidar(x3)
        hidden_stateThermo, outThermo = self.cnnexpertThermo(x4)

        #target = resultado de la funcion GPS a vector de posicion ver liinea 199
        #loss MSE en vez de Cross Entropy liinea 127

        outconcatgating = torch.cat([hidden_statergb, hidden_statedepth, hidden_stateLidar, hidden_stateThermo], dim = -1)
        outconcatexpertclassifier = torch.stack([outrgb, outdepth, outLidar, outThermo], dim = -1)
        gating = self.gatingnetwork(outconcatgating)
        #gating = gating.unsqueeze(1).expand(-1,2,-1) #32, 2, 3 era #32, 3 ####REVISAR AQUI AQUI ASDF
        gating = gating.unsqueeze(1)
        outfinal = outconcatexpertclassifier * gating
        return outfinal.sum(-1) ######## Regression instead of  classif. Check dim

    def train_dataloader(self):
        #trainset = torchvision.datasets.Kitti(root='./data', train=True, download=True, transform=self.transform)
        #trainset = ThermalDataset("~/Downloads/00")

        #dataset = MultiModalDataset(csv_file='multimodal.csv', root_dir='') /// root_dir is the folder with the images
        #trainset = MultiModalDataset(csv_file=dir_path + '/' + 'multimodal2.csv', root_dir=dir_path + '/'+'multimodalfolder', transform=self.transform)
        trainset = MultiModalDataset(csv_file=dir_path + '/' + 'multimodal.csv', root_dir=dir_path + '/'+'multimodalfolder', transform=self.transform)
        #print("dir_path ", dir_path)
        #trainset = MultiModalDataset(csv_file='multimodal.csv', root_dir='multimodalfolder', transform=self.transform)
        



        #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        #                                   download=True, transform=self.transform)
        



        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=10) #2
        return trainloader

    def test_dataloader(self):
        #testset = torchvision.datasets.Kitti(root='./data', train=False,
        #                                      download=True, transform=self.transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=10) #2
        return testloader

    ### optimizers and schedulers
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
    '''
    def training_step_before(self, train_batch, batch_idx):
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
        inputrgb, inputdepth, labels = train_batch
        #optimizer.zero_grad()
        batch_size = 8
        w, h = 128, 128
        #w=1392
        #h = 512

        labels = torch.rand(batch_size, 3)
        #inputrgb = torch.rand(batch_size, 3, w, h)
        #inputdepth = torch.rand(batch_size, 1, w, h)
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

        
        outputs = self(inputrgb, inputdepth, inputLidar, inputThermo) #forward(x1, x2, x3, x4)
        loss = self.criterion(outputs, labels)
        return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        outputs = self(images)
    
        _, predicted = torch.max(outputs.data, 1)

        #loss = F.cross_entropy(predicted, labels)
        loss = F.mse_loss(predicted, labels)

        return loss
