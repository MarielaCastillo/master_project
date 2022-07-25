import os 
from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint

from expert_rgb_module import LitModelEfficientNetRgb

dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    transform_rgb = transforms.Compose(
        [transforms.ToTensor(),
        #transforms.Resize((320, 960)),
        transforms.Resize((512, 640)),
        transforms.Normalize((0.5,), (0.5,))])
    
    model = LitModelEfficientNetRgb(1, transform_rgb)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints_rgb/')

    # trainer = Trainer(gpus=1, max_epochs=2, callbacks=[checkpoint_callback]))
    trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback])
    trainer.fit(model)



if __name__ == "__main__":
    main()