import math
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import register
from ray.rllib.env.multi_agent_env import  MultiAgentEnv

from core.chip import Chip
from core.reward_function import RewardFunction
import config
args = config.get_args()
rfunctions = RewardFunction()
'''
This env is to test our code,
The further the distance between the agents, the higher the reward.
agent acts simultaneously
'''
class Env_0(MultiAgentEnv):


    def __init__(self, config=None):
        super().__init__()
        self.steps = 0
        self.max_step = args.env_max_step
        self.num_qubits = args.num_qubits
        # define chip
        self.channel = 1  # RGB 图像
        self.chip: Chip = Chip(args.chip_rows, args.chip_cols)
        self.positions = []
        self.agents = self.possible_agents = [f"agent_{i+1}" for i in range(self.num_qubits)]

        self.obs_spaces = gym.spaces.Box(
            low=-2,
            high=128,
            shape=(args.chip_rows,args.chip_cols),
            dtype=np.int32
        )
        self.observation_spaces = {f"agent_{i+1}": self.obs_spaces for i in range(self.num_qubits)}
        self.action_spaces = {f"agent_{i+1}": gym.spaces.Discrete(4) for i in range(self.num_qubits)}


        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.chip.reset()
        self.default_distance = calculate_total_distance(self.chip._positions)
        self.last_distance = self.default_distance
        infos = {f'agent_{i + 1}':  self.default_distance for i in range(self.num_qubits)}
        return self._get_obs(),infos

    def _get_obs(self):
        obs = {f'agent_{i+1}': self.chip.state for i in range(self.num_qubits)}
        return obs

    def step(self, action_dict):
        self.steps += 1
        for i in range(1, self.num_qubits + 1):
            act = action_dict[f'agent_{i}']
            self.chip.move(i,act)

        rewards,distance = self.reward_function()

        terminateds = {"__all__": False} if self.steps <= self.max_step else {"__all__": True}
        truncated = {}
        infos = {f'agent_{i+1}':{'distance': distance} for i in range(self.num_qubits)}
        return self._get_obs(),rewards,terminateds,truncated,infos


    def reward_function(self):
        rewards = {}
        distance = calculate_total_distance(self.chip._positions)
        rf_name = f"rfv{args.rf_version}"
        function_to_call = getattr(rfunctions, rf_name, None)
        r = -1
        if callable(function_to_call):
            r = function_to_call(self, distance)
        else:
            print(f"Function {rf_name} does not exist.")
        for i in range(1, self.num_qubits+1):
            rewards.update({f'agent_{i}':r})
        return rewards,distance


def calculate_total_distance(coords) -> float:
    """
    计算所有二维坐标之间的距离总和
    :param coords: 坐标列表，每个元素为一个元组 (x, y)
    :return: 距离总和
    """
    total_distance = 0
    n = len(coords)

    # 遍历所有坐标对
    for i in range(1, n+1):
        for j in range(i, n):
            # 获取两个坐标点
            x1, y1 = coords[i]
            x2, y2 = coords[j]

            # 计算欧几里得距离
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 累加距离
            total_distance += distance

    return total_distance
if __name__ == '__main__':
    pass

