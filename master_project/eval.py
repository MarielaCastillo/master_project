import os 
from pytorch_lightning import Trainer
#from efnetLMmoduleseparated import LitModelEfficientNet
from efnetLMmoduleExpert1SS import LitModelEfficientNet
#from efnetLMmoduleExpert2SS import LitModelEfficientNet
#from efnetLMmoduleExpert3 import LitModelEfficientNet
import torchvision.transforms as transforms

import torch
from master_project.efnetLMmoduleExpert1SS import UNetExpert1

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    ################### Expert1 ###################################
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((320, 960)),
        transforms.Normalize((0.5,), (0.5,))])

    model = LitModelEfficientNet.load_from_checkpoint(batch_size=1, checkpoint_path="checkpoints/epoch=0-step=489.ckpt", transform=transform)
    model.eval()
    trainer = Trainer(accelerator="cpu",max_epochs=2)
    trainer.test(model=model)

if __name__ == "__main__":
    main()