from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from expert_rgb_module import LitModelEfficientNetRgb


def main():
    transform_rgb = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Resize((320, 960)),
         transforms.Resize((512, 640)),
         transforms.Normalize((0.5,), (0.5,))])

    chkpt_epochs=1

    model = LitModelEfficientNetRgb.load_from_checkpoint(batch_size=1,
                                                         # checkpoint_path="checkpoints_rgb/epoch=0-step=489.ckpt",
                                                         checkpoint_path="checkpoints_rgb/epoch=0-step=345.ckpt",
                                                         transform=transform_rgb, checkpoint_epochs=str(chkpt_epochs))
    model.eval()
    trainer = Trainer(gpus=1, max_epochs=1)
    # trainer = Trainer(accelerator="cpu", max_epochs=1)
    trainer.test(model=model)


if __name__ == "__main__":
    main()
