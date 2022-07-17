import os 
from pytorch_lightning import Trainer
#from efnetLMmoduleseparated import LitModelEfficientNet
from efnetLMmoduleExpert1SS import LitModelEfficientNet
#from efnetLMmoduleExpert2SS import LitModelEfficientNet
#from efnetLMmoduleExpert3 import LitModelEfficientNet
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((320, 960)),
        #transforms.Resize((200, 200)),
        transforms.Normalize((0.5,), (0.5,))])
    model = LitModelEfficientNet(1, transform) ##this number is batch size 
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/')

    #trainer = Trainer(gpus=1, max_epochs=2)
    trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback])
    #trainer = Trainer(accelerator="cpu",max_epochs=10)
    trainer.fit(model)

    # trainer.save_checkpoint("best_modelExp1.ckpt")
    #trainer.save_checkpoint("best_modelExp2.ckpt")
    #trainer.save_checkpoint("best_modelExp3.ckpt")


    # torch.save(model.state_dict(), dir_path+ '/' + 'models/modelExpert1.pt')
    #torch.save(model.state_dict(), dir_path+ '/' + 'models/modelExpert2.pt')
    #torch.save(model.state_dict(), dir_path+ '/' + 'models/modelExpert3.pt')

if __name__ == "__main__":
    main()