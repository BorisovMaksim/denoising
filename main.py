import hydra
from omegaconf import DictConfig
from train import train

# Parsing hydra config
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    main()
