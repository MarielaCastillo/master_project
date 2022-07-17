from pytorch_lightning import Trainer
#from efnetLMmoduleseparated import LitModelEfficientNet
from efnetLMmoduleExpert1 import LitModelEfficientNet
#from efnetLMmoduleExpert2 import LitModelEfficientNet
#from efnetLMmoduleExpert3 import LitModelEfficientNet
import torchvision.transforms as transforms

import torch

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((128, 128)),
        #transforms.Resize((200, 200)),
        transforms.Normalize((0.5,), (0.5,))])
    model = LitModelEfficientNet(8, transform) ##this number is batch size 

    #trainer = Trainer(gpus=1, max_epochs=2)
    trainer = Trainer(accelerator="cpu",max_epochs=2)
    #trainer = Trainer(accelerator="cpu",max_epochs=10)
    trainer.fit(model)
    trainer.save_checkpoint("best_modelExp1.ckpt")


    torch.save(model.state_dict(), dir_path+ '/' + 'models/modelExpert1.pt')
    #torch.save(model.state_dict(), dir_path+ '/' + 'models/modelExpert2.pt')
    #torch.save(model.state_dict(), dir_path+ '/' + 'models/modelExpert3.pt')

if __name__ == "__main__":
    main()