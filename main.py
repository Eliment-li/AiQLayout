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



if __name__ == "__main__":
    for version in range(1):
        print(version)