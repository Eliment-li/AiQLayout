import math
from copy import deepcopy
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
            low=-5,
            high=5,
            shape=(args.chip_rows + 2,args.chip_cols + 2), # +2 for padding
            dtype=np.int16
        )
        self.observation_spaces = {f"agent_{i+1}": self.obs_spaces for i in range(self.num_qubits)}
        self.action_spaces = {f"agent_{i+1}": gym.spaces.Discrete(5) for i in range(self.num_qubits)}

        self.player_now = 1  # index of the current agent

        self.agent_total_r = [0]* self.num_qubits

        self.max_total_r = [-np.inf] * self.num_qubits
        self.sw = [SlideWindow(50)] * self.num_qubits
        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.player_now = 1  # index of the current agent
        self.agent_total_r = [0] * self.num_qubits

        self.max_total_r = [-np.inf] * self.num_qubits
        self.sw = [SlideWindow(50)] * self.num_qubits

        self.chip.reset()
        self.init_dist =[calculate_distance_sum(i, self.chip.positions) for i in range(1, self.num_qubits + 1)]
        self.last_dist = [calculate_distance_sum(i, self.chip.positions) for i in range(1, self.num_qubits + 1)]
        self.max_dist = deepcopy(self.init_dist)

        infos = {f'agent_{i + 1}':  self.init_dist for i in range(self.num_qubits)}

        return self._get_obs(),infos

    def _get_obs(self):
        padded_state =  np.pad(self.chip.state, pad_width=1, mode='constant', constant_values=-5).astype(np.int16)
        obs = {f'agent_{self.player_now}': padded_state}
        return obs

    def step(self, action):
        self.steps += 1
        # print(f"step {self.steps} player {self.player_now} action {action}")
        act = action[f'agent_{self.player_now}']

        last_dist = calculate_distance_sum(self.player_now, self.chip.positions)
        self.chip.move(self.player_now,act)
        dist = calculate_distance_sum(self.player_now, self.chip.positions)
        rewards, distance = self.reward_function(dist=dist,last_dist=last_dist)
        self.last_dist[self.player_now - 1] = f'b{last_dist}-a{dist}'
        self.sw[self.player_now - 1].next(distance)
        terminateds = {"__all__": False} if self.steps < self.max_step else {"__all__": True}
        truncated = {}
        infos = {
                    f'agent_{self.player_now}':
                    {
                      'distance': self.last_dist[self.player_now - 1],
                        'max_total_r':self.max_total_r[self.player_now - 1]
                    }
                 }

        #make sure player_now is in the range of 1 to num_qubits
        self.player_now = ((self.player_now) % self.num_qubits) + 1
        return self._get_obs(),rewards,terminateds,truncated,infos


    def reward_function(self,dist,last_dist):
        # prepare rewrad function
        rf_name = f"rfv{args.rf_version}"
        rf_to_call = getattr(rfunctions, rf_name, None)
        assert callable(rf_to_call)

        rewards = {}
        p = self.player_now - 1
        _max_dist = self.max_dist[p]
        _max_total_r = self.max_total_r[p]
        _agent_total_r = self.agent_total_r[p]


        ##test
        # if self.player_now == 1:
        #     print(f"step {self.steps} player {self.player_now} action {act} distance {dist}")
        #     print(f'last_dist {self.last_dist}')
        ##
        if dist > _max_dist:
            if _max_dist == -np.inf:
                r = rf_to_call(init_dist=self.init_dist[p], last_dist=last_dist, dist=dist,
                               avg_dist=self.sw[p].current_avg)
            else:
                # 当 dist 首次出现这么大, 那么计算后的 total reward 也应该比之前所有的都大
                factor = (1 + (dist - _max_dist) / (_max_dist + 1))
                if factor < 1.1:
                    factor = 1.1
                if factor > 2:
                    factor = 2
                r = (_max_total_r - _agent_total_r * args.gamma) * factor
                if r < 0.1:
                    r = 0.1
                if r <0:
                    print(f'dist > _max_dist but r <0 {_max_total_r},{_agent_total_r},{dist},{_max_dist} ')
                    r = 0.1
                self.max_total_r[p] = _agent_total_r * args.gamma + r
            # update max_dist
            self.max_dist[p] = dist

        else:
            r = rf_to_call(init_dist=self.init_dist[p], last_dist=last_dist, dist=dist, avg_dist=self.sw[p].current_avg)

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

def calculate_distance_sum(player,positions) -> float:
    """
    参数:
    positions -- 包含坐标的列表，每个坐标是一个二元组(x, y)
    player -- 要计算的目标坐标的索引

    返回:
    第n个坐标与其他所有坐标的距离之和
    """
    assert len(positions) >= player >= 1, f"player {player} out of range"

    target_x, target_y = positions[player]
    total_distance = 0.0
    for i in range(1, len(positions)):
        x,y = positions[i]
        if i != player:  # 跳过自身
            distance = abs(x-target_x) + abs(y-target_y) #math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
            total_distance += distance

    return np.round(total_distance,4)


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
    act = [
        [1,2,1,2,1,3,4],
        [3,4,3,4, 3] ,
        [0,0,1,1,1] ,
        [2,2,3,3,3] ,
    ]
    print(env.last_dist)
    for i in range(5):
        print(env.last_dist)
        for k in range(4):
            warpped_act = {f'agent_{env.player_now}': act[k][i]}
            obs, reward, terminated, truncated, info = env.step(warpped_act)

            dist  = info[f'agent_{k+1}']['distance']
            print(dist)





