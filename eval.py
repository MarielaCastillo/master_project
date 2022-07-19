import os 
from pytorch_lightning import Trainer
#from efnetLMmoduleseparated import LitModelEfficientNet
from expert_rgb_module import LitModelEfficientNetRgb
from expert_thermo_module import LitModelEfficientNetThermo
#from efnetLMmoduleExpert2SS import LitModelEfficientNet
#from efnetLMmoduleExpert3 import LitModelEfficientNet
import torchvision.transforms as transforms
import torch

from expert_thermo_module import ScaleThermal

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    ################### Expert1 ###################################
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((320, 960)),
        transforms.Normalize((0.5,), (0.5,))])

    transform_thermo = transforms.Compose(
        [torch.tensor,
        ScaleThermal(max_value=30000),
        transforms.Resize((320, 960)),
        #transforms.Resize((200, 200)),
        transforms.Normalize((0.5,), (0.5,))])

    model = LitModelEfficientNetRgb.load_from_checkpoint(batch_size=1, checkpoint_path="checkpoints_rgb/epoch=0-step=489.ckpt", transform=transform)
    #model = LitModelEfficientNetThermo.load_from_checkpoint(batch_size=1, checkpoint_path="checkpoints_thermo/epoch=1-step=978.ckpt", transform=transform_thermo)
    model.eval()
    trainer = Trainer(accelerator="cpu",max_epochs=2)
    trainer.test(model=model)

if __name__ == "__main__":
    main()