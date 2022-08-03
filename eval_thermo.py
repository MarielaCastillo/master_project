import torch
from pytorch_lightning import Trainer
import torchvision.transforms as transforms

from expert_thermo_module import LitModelEfficientNetThermo
from expert_thermo_module import ScaleThermal
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    # Checkpoint file to use
    chkpt_path = "checkpoints_thermo/epoch=0-step=345.ckpt"
    
    # Get the epoch from name
    chkpt_epochs=chkpt_path.replace("checkpoints_thermo/epoch=", "")
    head, sep, tail = chkpt_epochs.partition('-')
    chkpt_epochs = head

    transform_thermo = transforms.Compose(
        [torch.tensor,
         ScaleThermal(max_value=30000),
         # transforms.Resize((320, 960)),
         transforms.Resize((512, 640)),
         transforms.Normalize((0.5,), (0.5,))])

    model = LitModelEfficientNetThermo.load_from_checkpoint(batch_size=1,
                                                            # checkpoint_path="checkpoints_thermo/epoch=1-step=978.ckpt",
                                                            checkpoint_path=chkpt_path, 
                                                            transform=transform_thermo, checkpoint_epochs=str(chkpt_epochs))
    logger = TensorBoardLogger("logs", name="expert_thermo_eval")

    model.eval()
    trainer = Trainer(gpus=1, max_epochs=1, logger=logger)
    # trainer = Trainer(accelerator="cpu", max_epochs=1, logger=logger)
    trainer.test(model=model)


if __name__ == "__main__":
    main()
