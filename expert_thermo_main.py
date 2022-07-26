import os 
import torch
from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint

from expert_thermo_module import LitModelEfficientNetThermo
from expert_thermo_module import ScaleThermal


dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    transform_thermo = transforms.Compose(
        [torch.tensor,
         ScaleThermal(max_value=30000),
         transforms.Resize((320, 960)),
         transforms.Normalize((0.5,), (0.5,))])

    model = LitModelEfficientNetThermo(1, transform_thermo)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints_thermo/')

    # trainer = Trainer(gpus=1, max_epochs=2, callbacks=[checkpoint_callback]))
    trainer = Trainer(accelerator="cpu", max_epochs=2, callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == "__main__":
    main()