import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms


####Model
class LitModel(pl.LightningModule):
    def __init__(self, batch_size, transform):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.criterion = nn.CrossEntropyLoss()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_dataloader(self):
        #trainset = torchvision.datasets.Kitti(root='./data', train=True,
        #                                      download=True, transform=self.transform)
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
