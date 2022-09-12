from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from end2end_module import LitModelEfficientNetFull
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

def main():
    epochs = 1
    batch_size = 1
    # Checkpoint file to use
    chkpt_path = "checkpoints_end2end/epoch=0-step=4129.ckpt"

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

    # Get the epoch from name
    chkpt_epochs=chkpt_path.replace("checkpoints_end2end/epoch=", "")
    head, sep, tail = chkpt_epochs.partition('-')
    chkpt_epochs = head

    model = LitModelEfficientNetFull.load_from_checkpoint(batch_size=batch_size,
                                                         # checkpoint_path="checkpoints_rgb/epoch=0-step=489.ckpt",
                                                         checkpoint_path=chkpt_path,
                                                         transform_rgb=transform_rgb, transform_thermo=transform_thermo,
                                                         checkpoint_epochs=str(chkpt_epochs))
    #logger = TensorBoardLogger("logs", name="full_eval")
        # logger = TensorBoardLogger("logs", name="full_eval")
    wandb_logger = WandbLogger(project="master_project", log_model="all")
    wandb_logger.log_hyperparams({"0name":"eval_end2end", "full_epochs":int(chkpt_epochs), "num_epochs":epochs})

    model.eval()
    trainer = Trainer(gpus=1, max_epochs=epochs, logger=wandb_logger)
    # trainer = Trainer(accelerator="cpu", max_epochs=epochs, logger=wandb_logger)
    trainer.test(model=model)


if __name__ == "__main__":
    main()
