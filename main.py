import hydra
from omegaconf import DictConfig, OmegaConf
from train import train


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == '__main__':
    main()
