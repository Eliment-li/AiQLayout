import hydra
from omegaconf import DictConfig, OmegaConf
import os

# 设置环境变量
os.environ["HYDRA_FULL_ERROR"] = "1"

# 运行你的代码

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()