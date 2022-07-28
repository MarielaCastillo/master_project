import os 
# from pytorch_lightning import 
from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.loggers import TensorBoardLogger



from expert_rgb_module import LitModelEfficientNetRgb

dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    transform_rgb = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Resize((320, 960)),
        transforms.Resize((512, 640)),
        transforms.Normalize((0.5,), (0.5,))])
    
    model = LitModelEfficientNetRgb(4, transform_rgb)  # batch_size
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints_rgb/')

    logger = TensorBoardLogger("tb_logs", name="my_model")

    # trainer = Trainer(gpus=3, max_epochs=2, callbacks=[checkpoint_callback])
    trainer = Trainer(gpus=3, max_epochs=2, callbacks=[checkpoint_callback], logger=logger)
    # trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback])
    # trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback], auto_lr_find=True)
    # trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback], logger=tb_logger)

    '''
    fig = lr_finder.plot(suggest=True)
    fig.show()
    model.hparams.learning_rate = lr_finder.suggestion()
    '''

    trainer.fit(model)



if __name__ == "__main__":
    main()