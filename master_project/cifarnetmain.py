from pytorch_lightning import Trainer
from lightningmoduleCifarNet import LitModelCifarNet
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    model = LitModelCifarNet(512, transform)
    #trainer = Trainer(gpus=1, max_epochs=2)
    trainer = Trainer(accelerator="cpu",max_epochs=2)
    trainer.fit(model)


if __name__ == "__main__":
    main()