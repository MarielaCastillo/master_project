import numpy as np
import torch 
import torch.nn as nn


from collections import OrderedDict

from torchvision import models

# class Architecture():
#     def __init__(self) -> None:
#         self.resnet = models.resnet18(pretrained=True)

#         print(self.resnet)


#https://github.com/gthparch/edgeBench/blob/master/pytorch/models/cifarnet.py
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()

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
            ('fc5', nn.Linear(192, 10))
        ]))

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)