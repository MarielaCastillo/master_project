from pytorch_lightning import Trainer
from lightningmoduleCifarNet import LitModelCifarNet
import torchvision.transforms as transforms

from master_project.lighningmodule import LitModel

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #model = LitModel(512, transform)
    model = LitModelCifarNet(512, transform)
    trainer = Trainer(gpus=1, max_epochs=2)
    trainer.fit(model)


if __name__ == "__main__":
    main()