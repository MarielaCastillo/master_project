import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms

from collections import OrderedDict

class CNNExpert(nn.Module):
    def __init__(self, inchannel, numclasses):
        super(CNNExpert, self).__init__()
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

        self.despuespool = nn.Sequential(OrderedDict([

            ('dropout4', nn.Dropout()),
            ('relu4', nn.ReLU(True)),
            ('fc5', nn.Linear(64, numclasses)) #2 instead of 10 porque 2 clases
        ]))

    def forward(self, input):
        hidden_state = self.antespool(input)
        return hidden_state, self.despuespool(hidden_state) #32x53824  and 1600x384
        
####Model
class LitModelCifarNet(pl.LightningModule):
    #https://github.com/gthparch/edgeBench/blob/master/pytorch/models/cifarnet.py#L23
    def __init__(self, batch_size, transform):
        super(LitModelCifarNet, self).__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.criterion = nn.CrossEntropyLoss()

        self.cnnexpertrgb = CNNExpert(3, 2)
        self.cnnexpertdepth = CNNExpert(1, 2) #hacer diferentes modelos para cada uno
        self.cnnexpertof = CNNExpert(2, 2)

        self.gatingnetwork = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64*3, 64)), #output 64
            ('relu3', nn.ReLU(True)),
            ('fc2', nn.Linear(64, 3)),   #output 2
            ('softmax', nn.Softmax(dim=1))
        ]))

        self.gatingnetworkGoogLeNetxxs = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(64, 128, kernel_size=3)), #output 128, kernel 3
            ('fc1', nn.Linear(128, 128)), #size 128
            ('relu3', nn.ReLU(True)),
            ('fc2', nn.Linear(128, 3)),   #size 2
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, x1, x2, x3):
        #outantespoolrgb = self.antespool(x1)
        #outantespooldepth = self.antespool(x2)
        #outantespoolof = self.antespool(x3)

        #outdespuespoolrgb = self.despuespool(outantespoolrgb)
        #outdespuespooldepth = self.despuespool(outantespooldepth)
        #outdespuespoolof = self.despuespool(outantespoolof)

        #outconcatgating = torch.cat([outantespoolrgb, outantespooldepth, outantespoolof], dim = -1)
        #outconcatexpertclassifier = torch.cat([outdespuespoolrgb, outdespuespooldepth, outdespuespoolof], dim = -1)

        hidden_statergb, outrgb = self.cnnexpertrgb(x1)
        hidden_statedepth, outdepth = self.cnnexpertdepth(x2)
        hidden_stateof, outof = self.cnnexpertof(x3)

        outconcatgating = torch.cat([hidden_statergb, hidden_statedepth, hidden_stateof], dim = -1)
        outconcatexpertclassifier = torch.stack([outrgb, outdepth, outof], dim = -1)

        gating = self.gatingnetwork(outconcatgating)
        gating = gating.unsqueeze(1).expand(-1,2,-1) #32, 2, 3 era #32, 3

        outfinal = outconcatexpertclassifier * gating
        #out = out.view(out.size(0), -1)
        return outfinal.sum(-1)

    def train_dataloader(self):
        #trainset = torchvision.datasets.Kitti(root='./data', train=True, download=True, transform=self.transform)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=2)
        return trainloader

    def test_dataloader(self):
        #testset = torchvision.datasets.Kitti(root='./data', train=False,
        #                                      download=True, transform=self.transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=2)
        return testloader

    ### optimizers and schedulers
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        #optimizer.zero_grad()
        batch_size = 8
        w, h = 128, 128
        labels = torch.randint(0, 2, [batch_size]).cuda() #high exclusive low inclusive
        inputrgb = torch.rand(batch_size, 3, w, h).cuda()
        inputdepth = torch.rand(batch_size, 1, w, h).cuda()
        inputof = torch.rand(batch_size, 2, w, h).cuda()
        outputs = self(inputrgb, inputdepth, inputof) #forward(x1, x2, x3)
        loss = self.criterion(outputs, labels)
        #loss.backward()
        return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        outputs = self(images)
    
        _, predicted = torch.max(outputs.data, 1)

        loss = F.cross_entropy(predicted, labels)

        return loss
