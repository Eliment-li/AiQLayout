import math
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import register
from ray.rllib.env.multi_agent_env import  MultiAgentEnv

from config import ConfigSingleton

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
        self.chip = None
        self.positions = []
        self.agents = self.possible_agents = [f"agent_{i}" for i in range(self.num_qubits)]

        self.obs_spaces = gym.spaces.Box(
            low=0,
            high=128,
            shape=(args.chip_size_h,args.chip_size_w),
            dtype=np.uint8
        )
        self.observation_spaces = {f"agent_{i}": self.obs_spaces for i in range(self.num_qubits)}
        self.action_spaces = {f"agent_{i}": gym.spaces.Discrete(4) for i in range(self.num_qubits)}


        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something

    def reset(self, *, seed=None, options=None):
        self.chip =  np.zeros((args.chip_size_w, args.chip_size_h), dtype=np.uint8)

        # 按顺序赋值
        value = 1  # 起始值
        for i in range(len(self.chip)):
            if value > self.num_qubits:
                break
            for j in range(len(self.chip[0])):
                self.chip[i][j] = value
                #position[i] = position of (i)_th qubit
                self.positions.append(
                    [i, j]
                )
                value += 1

        return self._get_obs(), {}

    def _get_obs(self):
        # obs = {
        #     "agent_1": self.chip,
        #     "agent_2": self.chip
        # }
        obs = {f'agent_{i}': self.chip for i in range(self.num_qubits)}
        return obs

    def step(self, action_dict):
        for i in range(self.num_qubits):
            move_i = action_dict[f'agent_{i}']
            self.move(i,move_i)

        rewards = self.reward_function()
        terminateds = {"__all__": False}
        truncated = {}
        infos = {}

        return self._get_obs(),rewards,terminateds,truncated,infos

    def reward(self):
        pass

    def move(self, player: int, act:int):
        old_pos = self.positions[player]
        if act == 0:
            new_pos = [old_pos[0], old_pos[1]+1]
        elif act == 1:
            new_pos = [old_pos[0], old_pos[1]-1]
        elif act == 2:
            new_pos = [old_pos[0]-1, old_pos[1]]
        elif act == 3:
            new_pos = [old_pos[0]+1, old_pos[1]]

        #if new_post out of matrix
        if (new_pos[0] < 0 or new_pos[0] >= args.chip_size_w or new_pos[1] < 0 or new_pos[1] >= args.chip_size_h or
                self.chip[new_pos[0]][new_pos[1]] != 0):
            return False
        else:
            try:
                self.positions[player] = new_pos
                self.chip[old_pos[0]][old_pos[1]] = 0
                self.chip[new_pos[0]][new_pos[1]] = player
            except Exception as e:
                pprint(f"Error: {e}")
                return False

            return True

    def reward_function(self):
        #todo call rfx

        #reward = (self.positions[0][0]-self.positions[1][0])**2 + (self.positions[0][1]-self.positions[1][1])**2
        reward = calculate_total_distance(self.positions)
        rewards = {f'agent_{i}':reward for i in range(self.num_qubits)}
        return rewards


def calculate_total_distance(coords):
    """
    计算所有二维坐标之间的距离总和

    :param coords: 坐标列表，每个元素为一个元组 (x, y)
    :return: 距离总和
    """
    total_distance = 0
    n = len(coords)

    # 遍历所有坐标对
    for i in range(n):
        for j in range(i + 1, n):
            # 获取两个坐标点
            x1, y1 = coords[i][0],coords[i][1]
            x2, y2 = coords[j][0],coords[j][1]

            # 计算欧几里得距离
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 累加距离
            total_distance += distance

    return total_distance
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

