import torch
from pytorch_lightning import Trainer
import torchvision.transforms as transforms

from expert_thermo_module import LitModelEfficientNetThermo
from expert_thermo_module import ScaleThermal


def main():
    transform_thermo = transforms.Compose(
        [torch.tensor,
         ScaleThermal(max_value=30000),
         transforms.Resize((320, 960)),
         transforms.Normalize((0.5,), (0.5,))])

    model = LitModelEfficientNetThermo.load_from_checkpoint(batch_size=1,
                                                            checkpoint_path="checkpoints_thermo/epoch=1-step=978.ckpt",
                                                            transform=transform_thermo)
    model.eval()
    trainer = Trainer(accelerator="cpu", max_epochs=2)
    trainer.test(model=model)


if __name__ == "__main__":
    main()
