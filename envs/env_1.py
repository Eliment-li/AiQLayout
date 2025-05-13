import math
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import register
from ray.rllib.env.multi_agent_env import  MultiAgentEnv

from config import ConfigSingleton
from core.chip import Chip, ChipAction
from core.reward_function import RewardFunction
from utils.calc_util import SlideWindow

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
            shape=(args.chip_rows + 2,args.chip_cols + 2), # +2 for padding
            dtype=np.int16
        )
        self.observation_spaces = {f"agent_{i+1}": self.obs_spaces for i in range(self.num_qubits)}
        self.action_spaces = {f"agent_{i+1}": gym.spaces.Discrete(5) for i in range(self.num_qubits)}

        self.player_now = 1  # index of the current agent

        self.agent_total_r = [0]* self.num_qubits
        self.max_dist = [-np.inf] * self.num_qubits
        self.max_total_r = [-np.inf] * self.num_qubits
        #TODO allocate each agent an slidewindow
        self.sw = SlideWindow(100)
        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.agent_total_r = [0,0,0,0]
        self.chip.reset()
        self.init_dist = calculate_total_distance(self.chip.positions)
        self.last_dist = self.init_dist
        infos = {f'agent_{i + 1}':  self.init_dist for i in range(self.num_qubits)}
        self.player_now = 1  # index of the current agent
        return self._get_obs(),infos

    def _get_obs(self):
        padded_state =  np.pad(self.chip.state, pad_width=1, mode='constant', constant_values=-9).astype(np.int16)
        obs = {f'agent_{self.player_now}': padded_state}
        return obs

    def step(self, action):
        self.steps += 1
        # print(f"step {self.steps} player {self.player_now} action {action}")
        act = action[f'agent_{self.player_now}']
        self.chip.move(self.player_now,act)

        rewards, distance = self.reward_function(act)
        self.last_dist = distance
        self.sw.next(distance)
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


    def reward_function(self,act):
        # prepare rewrad function
        rf_name = f"rfv{args.rf_version}"
        rf_to_call = getattr(rfunctions, rf_name, None)
        assert callable(rf_to_call)

        rewards = {}
        p = self.player_now - 1
        _max_dist = self.max_dist[p - 1]
        _max_total_r = self.max_total_r[p - 1]
        _agent_total_r = self.agent_total_r[p - 1]

        dist =calculate_distance_sum(self.player_now, self.chip.positions) #calculate_total_distance(self.chip.positions)

        if dist > _max_dist:
            if _max_dist == -np.inf:
                r = rf_to_call(init_dist=self.init_dist, last_dist=self.last_dist, dist=dist,
                               avg_dist=self.sw.current_avg)
            else:
                # 当 dist 首次出现这么大, 那么计算后的 total reward 也应该比之前所有的都大
                r = (_max_total_r - _agent_total_r * args.gamma) * (1 + (dist - _max_dist) / (_max_dist + 1))
                # update max_dist
                self.max_dist[p - 1] = dist
                self.max_total_r[p - 1] = _agent_total_r * args.gamma + r

            _max_dist = dist
        else:
            r = rf_to_call(init_dist=self.init_dist, last_dist=self.last_dist, dist=dist, avg_dist=self.sw.current_avg)

        for i in range(1, self.num_qubits + 1):
            if i == self.player_now:
                # update total reward for the current agent
                self.agent_total_r[i - 1] = self.agent_total_r[i - 1] * 0.99 + r
                # update max total r for the current agent
                if self.agent_total_r[i - 1] > self.max_total_r[i - 1]:
                    self.max_total_r[i - 1] = self.agent_total_r[i - 1]
                rewards.update({f'agent_{i}': r})
            else:
                rewards.update({f'agent_{i}': 0})

        return rewards,dist

def calculate_distance_sum( n,coords) -> float:
    """
    计算第n个坐标与其他所有坐标的距离之和

    参数:
    coords -- 包含坐标的列表，每个坐标是一个二元组(x, y)
    n -- 要计算的目标坐标的索引

    返回:
    第n个坐标与其他所有坐标的距离之和
    """
    if n < 0 or n >= len(coords):
        raise ValueError("索引n超出范围")
    target_x, target_y = coords[n]
    total_distance = 0.0

    for i in range(1, len(coords)):
        x,y = coords[i]
        if i != n:  # 跳过自身
            distance = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
            total_distance += distance

    return np.round(total_distance,3)


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
    env = Env_1()
    obs,infos = env.reset()
    print(obs)


