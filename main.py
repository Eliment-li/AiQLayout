import time

import gymnasium as gym
import hydra
import ray
from gymnasium import register
from omegaconf import DictConfig, OmegaConf
import os

from ray import tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME, EPISODE_RETURN_MEAN, ENV_RUNNER_RESULTS
from ray.tune import CLIReporter
from hydra import initialize, compose
from omegaconf import DictConfig

from envs import custom_env

args = None
# 手动初始化 Hydra
with initialize(config_path="conf", job_name="config",version_base="1.2"):
    # 加载配置文件
    confs = compose(config_name="config")
    args = OmegaConf.create({})
    for key in confs.keys():
        args = OmegaConf.merge(args, confs[key])

    for key, value in args.items():
        if str(value).lower() == 'none':
            # 如果值是字符串 'none'（忽略大小写），则替换为 None
            args[key] = None




if __name__ == "__main__":
    print(args)
    if args.num_gpus is None:
        print('none')
    # 解析命令行参数
    # args = parse_args()
    # print(OmegaConf.to_yaml(args))
    # print(args)
    # print(args.env_config)
    # print(args.env_config.key)
