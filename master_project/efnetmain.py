from pytorch_lightning import Trainer
from efnetLMmodule import LitModelEfficientNet
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5,), (0.5,))])
    model = LitModelEfficientNet(8, transform) ##this number is batch size 
    
    #check how to fix when total is not divisible by number of elements...
    #drop_batch

    #trainer = Trainer(gpus=1, max_epochs=2)
    trainer = Trainer(accelerator="cpu",max_epochs=2)
    trainer.fit(model)

if __name__ == "__main__":
    main()