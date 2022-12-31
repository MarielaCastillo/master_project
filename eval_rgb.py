from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from expert_rgb_module import LitModelEfficientNetRgb
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    # Checkpoint file to use
    chkpt_path = "checkpoints_rgb/epoch=49-step=17250.ckpt"

    # Get the epoch from name
    chkpt_epochs=chkpt_path.replace("checkpoints_rgb/epoch=", "")
    head, sep, tail = chkpt_epochs.partition('-')
    chkpt_epochs = head

    transform_rgb = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Resize((320, 960)),
         transforms.Resize((512, 640)),
         transforms.Normalize((0.5,), (0.5,))])

    model = LitModelEfficientNetRgb.load_from_checkpoint(batch_size=1,
                                                         # checkpoint_path="checkpoints_rgb/epoch=0-step=489.ckpt",
                                                         checkpoint_path=chkpt_path,
                                                         transform=transform_rgb, checkpoint_epochs=str(chkpt_epochs))
    logger = TensorBoardLogger("logs", name="expert_rgb_eval")

    model.eval()
    trainer = Trainer(gpus=1, max_epochs=1, logger=logger)
    # trainer = Trainer(accelerator="cpu", max_epochs=1, logger=logger)
    trainer.test(model=model)


if __name__ == "__main__":
    main()
