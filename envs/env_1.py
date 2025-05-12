import math
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import register
from ray.rllib.env.multi_agent_env import  MultiAgentEnv

from config import ConfigSingleton
from core.chip import Chip, ChipAction
from core.reward_function import RewardFunction
args = ConfigSingleton().get_args()
rfunctions = RewardFunction()
'''
This env is to test our code,
The further the distance between the agents, the higher the reward.
the agent acts one by  one
'''
class Env_1(MultiAgentEnv):


    def __init__(self, config=None):
        super().__init__()
        self.steps = 0
        self.num_qubits = args.num_qubits
        self.max_step = args.env_max_step * self.num_qubits
        # define chip
        self.channel = 1  # RGB 图像
        self.chip = Chip(args.chip_rows, args.chip_cols)
        self.positions = []
        self.agents = self.possible_agents = [f"agent_{i+1}" for i in range(self.num_qubits)]

        self.obs_spaces = gym.spaces.Box(
            low=-10,
            high=16,
            shape=(args.chip_rows,args.chip_cols),
            dtype=np.int16
        )
        self.observation_spaces = {f"agent_{i+1}": self.obs_spaces for i in range(self.num_qubits)}
        self.action_spaces = {f"agent_{i+1}": gym.spaces.Discrete(5) for i in range(self.num_qubits)}

        self.player_now = 1  # index of the current agent
        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.chip.reset()
        self.default_distance = calculate_total_distance(self.chip._positions)
        self.last_distance = self.default_distance
        infos = {f'agent_{i + 1}':  self.default_distance for i in range(self.num_qubits)}
        self.player_now = 1  # index of the current agent
        return self._get_obs(),infos

    def _get_obs(self):
        padded_state =  np.pad(self.chip.state, pad_width=1, mode='constant', constant_values=-9)
        obs = {f'agent_{self.player_now}': padded_state}
        return obs

    def step(self, action):
        self.steps += 1
        # print(f"step {self.steps} player {self.player_now} action {action}")
        act = action[f'agent_{self.player_now}']
        self.chip.move(self.player_now,act)
        if act == ChipAction.STAY:
            rewards = 0
            distance = self.last_distance
        else:
            rewards, distance = self.reward_function()

        terminateds = {"__all__": False} if self.steps <= self.max_step else {"__all__": True}
        truncated = {}
        infos = {
                    f'agent_{self.player_now}':
                    {
                      'distance': distance
                    }
                 }

        #make sure player_now is in the range of 1 to num_qubits
        self.player_now = ((self.player_now) % self.num_qubits) + 1
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
            if i == self.player_now:
                rewards.update({f'agent_{i}':r})
            else:
                rewards.update({f'agent_{i}':0})
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
