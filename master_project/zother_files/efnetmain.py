from pytorch_lightning import Trainer
#from efnetLMmodule import LitModelEfficientNet
from efnetLMmoduleSS import LitModelEfficientNet
import torchvision.transforms as transforms

from efficientnet_pytorch import EfficientNet
import torch.nn as nn

import torch

import segmentation_models_pytorch as smp

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    ### Expert1 #############################################################################
    model1 = UNetExpert1(3)
    checkpoint = torch.load("best_modelExp1.ckpt")

    model_weights = checkpoint["state_dict"]

    # update keys by dropping `auto_encoder.`
    for key in list(model_weights):
        model_weights[key.replace("cnnexpertRGB.", "")] = model_weights.pop(key)
        
    model1.load_state_dict(model_weights)
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


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5,), (0.5,))])
    #model = LitModelEfficientNet(8, transform) ##this number is batch size 
    #model = LitModelEfficientNet(8, transform, model1, model2) ##this number is batch size 
    model = LitModelEfficientNet(8, transform, model1) 
    
    #check how to fix when total is not divisible by number of elements...
    #drop_batch

    #trainer = Trainer(gpus=1, max_epochs=2)
    print(dir_path)
    #trainer = Trainer(accelerator="cpu",max_epochs=2, default_root_dir=dir_path+ '/' + 'models')
    trainer = Trainer(accelerator="cpu",max_epochs=2)
    trainer.fit(model)
    #trainer.save_checkpoint("best_model.ckpt")


    #torch.save(model.state_dict(), dir_path+ '/' + 'models/model.pt')

class UNetExpert1(nn.Module):
    def __init__(self, numclasses):
        super().__init__()
        self.model = smp.Unet('efficientnet-b4', classes=numclasses, activation='softmax')

    def forward(self, input):
        hidden_state = self.model.encoder(input)
        decoder_output = self.model.decoder(*hidden_state)

        masks = self.model.segmentation_head(decoder_output)
        if self.model.classification_head is not None:
            labels = self.model.classification_head(hidden_state[-1])
            return hidden_state, masks, labels
        return hidden_state, masks

class CNNExpert1(nn.Module):
    def __init__(self, inchannel, numclasses):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b4', in_channels=inchannel)
        self.disable_gradients(self.efficient_net)
        self.e2h = nn.Linear(1792, 64)
        self.hidden_dim = 64
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=256),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=numclasses)
        )

    def disable_gradients(self, model):
        # update the pretrained model
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, input):
        features = self.efficient_net.extract_features(input)
        hidden_state = self.efficient_net._avg_pooling(features)
        hidden_state = hidden_state.flatten(start_dim=1)
        hidden_state = self.e2h(hidden_state)
        return hidden_state, self.fc(hidden_state)

if __name__ == "__main__":
    main()