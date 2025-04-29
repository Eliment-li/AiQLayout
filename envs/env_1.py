import math
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import register
from ray.rllib.env.multi_agent_env import  MultiAgentEnv

from config import ConfigSingleton
from core.chip import Chip

args = ConfigSingleton().get_args()

'''
actions:

hold on

use nearest path
use second-optimal pathways
use third-optimal pathways
'''
class Env_1(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()
        self.num_qubits = args.num_qubits
        # define chip
        self.channel = 1  # RGB 图像
        self.chip: Chip = None
        self.positions = []
        self.agents = self.possible_agents = [f"agent_{i+1}" for i in range(self.num_qubits)]

        self.obs_spaces = gym.spaces.Box(
            low=-2,
            high=128,
            shape=(args.chip_size_h,args.chip_size_w),
            dtype=np.int32
        )
        self.observation_spaces = {f"agent_{i+1}": self.obs_spaces for i in range(self.num_qubits)}
        self.action_spaces = {f"agent_{i+1}": gym.spaces.Discrete(4) for i in range(self.num_qubits)}


        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something

    def reset(self, *, seed=None, options=None):
        self.chip =  Chip()
        self.chip.reset()

        return self._get_obs(), {}

    def _get_obs(self):
        # obs = {
        #     "agent_1": self.chip,
        #     "agent_2": self.chip
        # }
        obs = {f'agent_{i+1}': self.chip.state for i in range(self.num_qubits)}
        return obs

    def step(self, action_dict):
        for i in range(self.num_qubits):
            act = action_dict[f'agent_{i+1}']
            self.chip.move(i+1,act)

        rewards = self.reward_function()
        terminateds = {"__all__": False}
        truncated = {}
        infos = {}

        return self._get_obs(),rewards,terminateds,truncated,infos

    def reward(self):
        pass

    def reward_function(self):
        #todo call rfx
        #reward = (self.positions[0][0]-self.positions[1][0])**2 + (self.positions[0][1]-self.positions[1][1])**2
        path_len = self.chip.all_path_len()
        rewards = {}
        for i in range(self.num_qubits):
            r = (5 - path_len[i])
            rewards.update({f'agent_{i+1}':r})
        return rewards



if __name__ == '__main__':
    # env = Env_1()
    # obs,_ = env.reset()
    # pprint(obs)
    print(gym.spaces.Box(
        low=0,
        high=128,
        shape=(2, 3),
        dtype=np.uint8
    ).sample())

