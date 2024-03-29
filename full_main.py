import torch
from pytorch_lightning import Trainer
from expert_rgb_module import LitModelEfficientNetRgb
from expert_thermo_module import LitModelEfficientNetThermo
from full_module import LitModelEfficientNetFull
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from expert_thermo_module import ScaleThermal

from pytorch_lightning.loggers import WandbLogger

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    transform_rgb = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((512, 640)),
        transforms.Normalize((0.5,), (0.5,))])

    transform_thermo = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((512, 640)),
        transforms.Normalize((0.5,), (0.5,))])

    model_rgb = LitModelEfficientNetRgb.load_from_checkpoint(
        batch_size=1,
        # checkpoint_path="checkpoints_rgb/epoch=0-step=489.ckpt",  # Freiburg Thermal 
        checkpoint_path="checkpoints_rgb/epoch=499-step=129500.ckpt",
        # checkpoint_path="checkpoints_rgb/epoch=49-step=17250.ckpt",
        transform=transform_rgb)
    model_thermo = LitModelEfficientNetThermo.load_from_checkpoint(
        batch_size=1,
        # checkpoint_path="checkpoints_thermo/epoch=1-step=978.ckpt",
        checkpoint_path="checkpoints_thermo/epoch=499-step=129500.ckpt",
        transform=transform_thermo)
    expert_rgb = model_rgb.cnnexpert
    expert_thermo = model_thermo.cnnexpert

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints_full/')

    logger = TensorBoardLogger("logs", name="full")
    
    wandb_logger = WandbLogger(project="master_project3", log_model="all")
    # wandb_logger.log_hyperparams({})

    # trainer = Trainer(accelerator="cpu", max_epochs=1, callbacks=[checkpoint_callback], logger=logger)
    # trainer = Trainer(accelerator="cpu", max_epochs=1, callbacks=[checkpoint_callback], logger=wandb_logger)
    # trainer = Trainer(gpus=1, max_epochs=1, callbacks=[checkpoint_callback], logger=logger)
    trainer = Trainer(gpus=1, max_epochs=1, callbacks=[checkpoint_callback], logger=wandb_logger)

    model = LitModelEfficientNetFull(1, transform_rgb=transform_rgb, transform_thermo=transform_thermo,
                                     model1=expert_rgb, model2=expert_thermo)
    trainer.fit(model)


if __name__ == "__main__":
    main()
