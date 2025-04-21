import gymnasium as gym

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import hydra
from omegaconf import DictConfig, OmegaConf
import os
# 设置环境变量
# os.environ["HYDRA_FULL_ERROR"] = "1"

# 运行你的代码

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(args : DictConfig):
    pass

def evaluate():
    pass

if __name__ == '__main__':
    pass