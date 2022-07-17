from pytorch_lightning import Trainer
from efnetLMmoduleExpert1SS import LitModelEfficientNet
from efnetLMmoduleSS import LitModelEfficientNetFull
import torchvision.transforms as transforms

from pytorch_lightning.callbacks import ModelCheckpoint

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((320, 960)),
        transforms.Normalize((0.5,), (0.5,))])
    model_rgb = LitModelEfficientNet.load_from_checkpoint(batch_size=1, checkpoint_path="checkpoints/epoch=0-step=489.ckpt", transform=transform)
    model_thermo = LitModelEfficientNet.load_from_checkpoint(batch_size=1, checkpoint_path="checkpoints/epoch=1-step=978.ckpt", transform=transform)
    expert_rgb = model_rgb.cnnexpertRGB
    expert_thermo = model_thermo.cnnexpertRGB

    checkpoint_callback = ModelCheckpoint(dirpath='models_with_pretrained_experts/')
    trainer = Trainer(accelerator="cpu",max_epochs=2, callbacks=[checkpoint_callback])
    model = LitModelEfficientNetFull(1, transform, expert_rgb, expert_thermo) 
    trainer.fit(model)
    

    #trainer = Trainer(gpus=1, max_epochs=2)
    #trainer = Trainer(accelerator="cpu",max_epochs=2, default_root_dir=dir_path+ '/' + 'models')

    
    #trainer.save_checkpoint("best_model.ckpt")


    #torch.save(model.state_dict(), dir_path+ '/' + 'models/model.pt')

if __name__ == "__main__":
    main()