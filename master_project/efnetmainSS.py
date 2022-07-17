from pytorch_lightning import Trainer
#from efnetLMmodule import LitModelEfficientNet
from efnetLMmoduleExpert1SS import LitModelEfficientNet
from efnetLMmoduleSS import LitModelEfficientNetFull
import torchvision.transforms as transforms

from efficientnet_pytorch import EfficientNet
import torch.nn as nn

import torch

import segmentation_models_pytorch as smp

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((320, 960)),
        transforms.Normalize((0.5,), (0.5,))])
    model1 = LitModelEfficientNet.load_from_checkpoint(batch_size=1, checkpoint_path="checkpoints/epoch=0-step=489.ckpt", transform=transform)
    model2 = LitModelEfficientNet.load_from_checkpoint(batch_size=1, checkpoint_path="checkpoints/epoch=1-step=978.ckpt", transform=transform)
    expert_rgb = model1.cnnexpertRGB
    expert_thermo = model2.cnnexpertRGB
    '''
    ### Expert2 #############################################################################
    model2 = CNNExpert1(1,3)
    checkpoint = torch.load("best_modelExp2.ckpt")

    model_weights = checkpoint["state_dict"]

    # update keys by dropping `auto_encoder.`
    for key in list(model_weights):
        model_weights[key.replace("cnnexpertDepth.", "")] = model_weights.pop(key)
        
    model2.load_state_dict(model_weights)
    '''



    #model = LitModelEfficientNet(8, transform) ##this number is batch size 
    #model = LitModelEfficientNet(8, transform, model1, model2) ##this number is batch size 
    model = LitModelEfficientNetFull(1, transform, expert_rgb, expert_thermo) 
    
    #check how to fix when total is not divisible by number of elements...
    #drop_batch

    #trainer = Trainer(gpus=1, max_epochs=2)
    print(dir_path)
    #trainer = Trainer(accelerator="cpu",max_epochs=2, default_root_dir=dir_path+ '/' + 'models')
    trainer = Trainer(accelerator="cpu",max_epochs=2)
    trainer.fit(model)
    #trainer.save_checkpoint("best_model.ckpt")


    #torch.save(model.state_dict(), dir_path+ '/' + 'models/model.pt')

if __name__ == "__main__":
    main()