import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms

from collections import OrderedDict

####Model
class LitModelCifarNet(pl.LightningModule):
    #https://github.com/gthparch/edgeBench/blob/master/pytorch/models/cifarnet.py#L23
    def __init__(self, batch_size, transform):
        super(LitModelCifarNet, self).__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.criterion = nn.CrossEntropyLoss()

        self.feature = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=5)),
            ('batch1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=5)),
            ('batch2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(1600, 384)),
            ('dropout3', nn.Dropout()),
            ('relu3', nn.ReLU(True)),
            ('fc4', nn.Linear(384, 192)),
            ('dropout4', nn.Dropout()),
            ('relu4', nn.ReLU(True)),
            ('fc5', nn.Linear(192, 10)) #2 instead of 10 porque 2 clases
        ]))

        self.gatingnetwork = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64, 64)), #output 64
            ('relu3', nn.ReLU(True)),
            ('fc1', nn.Linear(64, 2)),   #output 2
            ('softmax', nn.Softmax(dim=1))
        ]))

        self.gatingnetworkGoogLeNetxxs = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(64, 128, kernel_size=3)), #output 128, kernel 3
            ('fc1', nn.Linear(128, 128)), #size 128
            ('relu3', nn.ReLU(True)),
            ('fc1', nn.Linear(128, 2)),   #size 2
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

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
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        #loss.backward()
        return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        outputs = self(images)
    
        _, predicted = torch.max(outputs.data, 1)

        loss = F.cross_entropy(predicted, labels)

        return loss
