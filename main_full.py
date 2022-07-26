import torch
from pytorch_lightning import Trainer
from expert_rgb_module import LitModelEfficientNetRgb
from expert_thermo_module import LitModelEfficientNetThermo
from main_module import LitModelEfficientNetFull
import torchvision.transforms as transforms

from pytorch_lightning.callbacks import ModelCheckpoint

from expert_thermo_module import ScaleThermal

import os 
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
         transforms.Normalize((0.5,), (0.5,))])

    model_rgb = LitModelEfficientNetRgb.load_from_checkpoint(
        batch_size=1,
        checkpoint_path="checkpoints_rgb/epoch=0-step=489.ckpt",
        transform=transform_rgb)
    model_thermo = LitModelEfficientNetThermo.load_from_checkpoint(
        batch_size=1,
        checkpoint_path="checkpoints_thermo/epoch=1-step=978.ckpt",
        transform=transform_thermo)
    expert_rgb = model_rgb.cnnexpert
    expert_thermo = model_thermo.cnnexpert

    checkpoint_callback = ModelCheckpoint(dirpath='models_with_pretrained_experts/')
    trainer = Trainer(accelerator="cpu", max_epochs=2, callbacks=[checkpoint_callback])

    # trainer = Trainer(gpus=3, max_epochs=2, callbacks=[checkpoint_callback])
    model = LitModelEfficientNetFull(1, transform_rgb=transform_rgb, transform_thermo=transform_thermo,
                                     model1=expert_rgb, model2=expert_thermo)
    trainer.fit(model)

    # trainer = Trainer(gpus=1, max_epochs=2)
    # trainer = Trainer(accelerator="cpu",max_epochs=2, default_root_dir=dir_path+ '/' + 'models')

    # trainer.save_checkpoint("best_model.ckpt")
    # torch.save(model.state_dict(), dir_path+ '/' + 'models/model.pt')


if __name__ == "__main__":
    main()
