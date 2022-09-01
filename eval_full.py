from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from expert_rgb_module import LitModelEfficientNetRgb
from expert_thermo_module import LitModelEfficientNetThermo
from full_module import LitModelEfficientNetFull
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    # Checkpoint file to use
    # chkpt_path = "checkpoints_full/epoch=0-step=4129.ckpt"
    chkpt_path = "checkpoints_full/epoch=39-step=165160.ckpt"

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
        # checkpoint_path="checkpoints_rgb/epoch=0-step=345.ckpt",
        # checkpoint_path="checkpoints_rgb/epoch=49-step=17250.ckpt",
        checkpoint_path="checkpoints_rgb/epoch=499-step=129500.ckpt",
        transform=transform_rgb)
    model_thermo = LitModelEfficientNetThermo.load_from_checkpoint(
        batch_size=1,
        # checkpoint_path="checkpoints_thermo/epoch=1-step=978.ckpt",
        # checkpoint_path="checkpoints_thermo/epoch=0-step=345.ckpt",
        checkpoint_path="checkpoints_thermo/epoch=499-step=129500.ckpt",
        transform=transform_thermo)

    # Get the epoch from name
    chkpt_epochs=chkpt_path.replace("checkpoints_full/epoch=", "")
    head, sep, tail = chkpt_epochs.partition('-')
    chkpt_epochs = head

    model = LitModelEfficientNetFull.load_from_checkpoint(batch_size=1,
                                                         # checkpoint_path="checkpoints_rgb/epoch=0-step=489.ckpt",
                                                         checkpoint_path=chkpt_path,
                                                         transform_rgb=transform_rgb, transform_thermo=transform_thermo,
                                                         model1=model_rgb.cnnexpert, model2 = model_thermo.cnnexpert,
                                                         checkpoint_epochs=str(chkpt_epochs))
    logger = TensorBoardLogger("logs", name="full_eval")

    model.eval()
    trainer = Trainer(gpus=1, max_epochs=1, logger=logger)
    # trainer = Trainer(accelerator="cpu", max_epochs=1, logger=logger)
    trainer.test(model=model)


if __name__ == "__main__":
    main()
