import os 
# from pytorch_lightning import 
from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from end2end_module import LitModelEfficientNetFull

dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    transform_rgb = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Resize((320, 960)),
        transforms.Resize((512, 640)),
        transforms.Normalize((0.5,), (0.5,))])

    transform_thermo = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Resize((320, 960)),
        transforms.Resize((512, 640)),
        transforms.Normalize((0.5,), (0.5,))])

    model = LitModelEfficientNetFull(1, transform_rgb=transform_rgb, transform_thermo=transform_thermo)

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints_end2end/')

    logger = TensorBoardLogger("logs", name="end2end")

    # trainer = Trainer(gpus=3, max_epochs=2, callbacks=[checkpoint_callback])
    # trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback])
    # trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback], auto_lr_find=True)

    trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback], logger=logger)
    # trainer = Trainer(gpus=3, max_epochs=1, callbacks=[checkpoint_callback], logger=logger)

    trainer.fit(model)


if __name__ == "__main__":
    main()