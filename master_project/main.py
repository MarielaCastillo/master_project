import numpy
import torch
import torch.nn.functional as F

from torch import nn
from torchmetrics import iOU #is there anothe way to do this?


def main(data_dir,
         torch_model,
         num_epochs=10,
         batch_size=50,
         learning_rate=0.001, ## paper says 0
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.Adam,
         data_augmentations=None,
         save_model_str=None,
         use_all_data_to_train=False,
         exp_name=''):

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # C dimensional vector
    C = 2
    # one-hot encoding
    F.one_hot(torch.arange(0, 5) % 3, num_classes=10)

    ###
    #CifarNet

    
    #Softmax
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)

    s1 = F.softmax(x, dim=0)

    #Fast R-CNN
    self.conv = torch.nn.Conv2d(in_channels=3, out_channels=self.numfilters, kernel_size=self.kernel, padding=self.padding)

    #Dropout in FCC
    self.dropout = nn.Dropout(0.25)

    #CrossEntropy Loss
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss}

    ####or CE loss 2
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()

    # SGD
    opti_dict = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam} 

    #RGB

    #Depth

    #Optical Flow
    ## OpenCV

    #IoU
    target = torch.randint(0, 2, (10, 25, 25))
    pred = torch.tensor(target)
    pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
    iou = IoU(num_classes=2)
    iou(pred, target)

    ###Gating2
    self.linear1 = torch.nn.Linear(64, 3)
    self.linear2 = torch.nn.Linear(64, 3)
