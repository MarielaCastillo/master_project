import os 
import torch
from pytorch_lightning import Trainer
#from efnetLMmoduleseparated import LitModelEfficientNet
from efnetLMmoduleExpert1SS import LitModelEfficientNetRgb
from efnetLMmoduleExpert2SS import LitModelEfficientNetThermo
#from efnetLMmoduleExpert3 import LitModelEfficientNet
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from efnetLMmoduleExpert2SS import ScaleThermal


dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    transform_rgb = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((320, 960)),
        transforms.Normalize((0.5,), (0.5,))])

    transform_thermo = transforms.Compose(
        [torch.tensor,
        ScaleThermal(max_value=30000),
        transforms.Resize((320, 960)),
        #transforms.Resize((200, 200)),
        transforms.Normalize((0.5,), (0.5,))])
    
    #model = LitModelEfficientNetRgb(1, transform_rgb)
    model = LitModelEfficientNetThermo(1, transform_thermo)
    
    #checkpoint_callback = ModelCheckpoint(dirpath='checkpoints_rgb/')
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints_thermo/')

    #trainer = Trainer(gpus=1, max_epochs=2)
    trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == "__main__":
    main()