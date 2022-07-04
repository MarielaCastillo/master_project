import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="config", config_name="train")
def my_app(cfg : DictConfig) -> None:
    #print(cfg)
    print(cfg.persona.nombre)
    print(cfg["persona"])

if __name__ == "__main__":
    my_app()