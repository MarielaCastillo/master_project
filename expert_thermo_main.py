import os 
import torch
from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from expert_thermo_module import LitModelEfficientNetThermo
from expert_thermo_module import ScaleThermal

dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    '''
    transform_thermo = transforms.Compose(
        [torch.tensor,
         ScaleThermal(max_value=30000),
         # transforms.Resize((320, 960)),
         transforms.Resize((512, 640)),
         transforms.Normalize((0.5,), (0.5,))])
    '''
    transform_thermo = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Resize((320, 960)),
        transforms.Resize((512, 640)),
        transforms.Normalize((0.5,), (0.5,))])
    

    model = LitModelEfficientNetThermo(4, transform_thermo)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints_thermo/')

    logger = TensorBoardLogger("logs", name="expert_thermo")

    # trainer = Trainer(accelerator="cpu", max_epochs=1, callbacks=[checkpoint_callback], logger=logger)
    trainer = Trainer(gpus=3, max_epochs=1, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model)


if __name__ == "__main__":
    main()