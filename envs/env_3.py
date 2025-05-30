import math
from copy import deepcopy
from pprint import pprint
import  random

import gymnasium as gym
import numpy as np
from gymnasium import register
from numpy import dtype
from ray.rllib.env.multi_agent_env import  MultiAgentEnv

from config import ConfigSingleton
from core.chip import Chip, ChipAction
from core.reward_function import RewardFunction
from core.reward_scaling import RewardScaling
from utils.calc_util import SlideWindow
from utils.position import positionalencoding2d

args = ConfigSingleton().get_args()
rfunctions = RewardFunction()
'''
new feat:
use positional encoding 

This env is to guide qubits to get closer to magic state
the agent acts one by one
'''
class Env_3(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()
        self.steps = 0
        self.num_qubits = args.num_qubits
        self.max_step = args.env_max_step * self.num_qubits
        # define chip
        self.chip = Chip(args.chip_rows, args.chip_cols)
        self.positions = []
        self.agents = self.possible_agents = [f"agent_{i+1}" for i in range(self.num_qubits)]
        self.obs_spaces = gym.spaces.Box(
            low=-6,
            high=6,
            shape=(4+1,args.chip_rows,args.chip_cols),
            dtype=np.float32,
        )
        self.observation_spaces = {f"agent_{i+1}": self.obs_spaces for i in range(self.num_qubits)}
        self.action_spaces = {f"agent_{i+1}": gym.spaces.Discrete(6) for i in range(self.num_qubits)}

        self.player_now = 1  # index of the current agent

        self.agent_total_r = [0]* self.num_qubits

        self.max_total_r = [-np.inf] * self.num_qubits
        self.sw = [SlideWindow(50)] * self.num_qubits
        #use config from outside
        # if config.get("sheldon_cooper_mode"):
        #     #do something
        self.pe = positionalencoding2d(9,9,4)

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.player_now = 1  # index of the current agent
        self.done = [False] * self.num_qubits
        self.agent_total_r = [0] * self.num_qubits

        self.max_total_r = [-np.inf] * self.num_qubits
        self.sw = [SlideWindow(50)] * self.num_qubits

        self.chip.reset()
        self.init_dist =[self.distance_to_m(i) for i in range(1, self.num_qubits + 1)]
        self.dist_rec = [[] for i in range(1, self.num_qubits + 1)]
        self.min_dist = deepcopy(self.init_dist)

        self.rs = RewardScaling(shape=1, gamma=0.9)
        self.rs.reset()

        infos = {f'agent_{i + 1}':  self.init_dist for i in range(self.num_qubits)}

        return self._get_obs(),infos

    def _get_obs(self):
        #padded_state =  np.pad(self.chip.channel_state, pad_width=1, mode='constant', constant_values=-5).astype(np.int16)

        repeat_state = np.repeat(self.chip.state[np.newaxis, :, :], 4, axis=0)
        obs = repeat_state + self.pe
        pm =np.expand_dims(self.chip.position_mask(self.player_now), axis=0)
        obs = np.concatenate((obs,pm),axis=0)  # (4, rows, cols) -> (4+1, rows, cols)
        ret = {
            f'agent_{self.player_now}':obs
        }
        return ret

    def step(self, action):
        self.steps += 1
        terminateds = self.is_terminated()

        # print(f"step {self.steps} player {self.player_now} action {action}")
        act = action[f'agent_{self.player_now}']
        if act == ChipAction.Done.value or  self.done[self.player_now - 1]:
            #print(f'player {self.player_now} done at step {self.steps}')
            self.done[self.player_now - 1] = True
            dist =  last_dist = self.distance_to_m(self.player_now)
            rewards= {f'agent_{self.player_now}': self.rs(0)[0]}
        else:
            last_dist = self.distance_to_m(self.player_now)
            self.chip.move(player=self.player_now,act=act)
            dist = self.distance_to_m(self.player_now)
            rewards = self.reward_function(dist=dist,last_dist=last_dist)

        self.dist_rec[self.player_now - 1] = f'{last_dist}->{dist}'
        self.sw[self.player_now - 1].next(dist)

        truncated = {}
        infos = {
                    f'agent_{self.player_now}':
                    {
                        'distance': self.dist_rec[self.player_now - 1],
                        'max_total_r':self.max_total_r[self.player_now - 1]
                    }
                 }

        if dist <=1:
            self.done[self.player_now - 1] = True

        self.player_now = ((self.player_now) % self.num_qubits) + 1

        # switch to next player
        if not np.all(self.done):
            while self.done[self.player_now - 1]:
                self.player_now = ((self.player_now) % self.num_qubits) + 1

        return self._get_obs(),rewards,terminateds,truncated,infos

    def is_terminated(self):

        if self.steps >= self.max_step:
           terminateds = {"__all__": True}
        elif np.all(self.done):
           terminateds = {"__all__": True}
        else:
            terminateds = {"__all__": False}
            # for i in range(len(self.done)):
            #     if self.done[i]:
            #         terminateds.update({f'agent_{i + 1}': True})
            #     else:
            #         terminateds.update({'__all__': False})
            #         terminateds.update({f'agent_{i + 1}': False})

        return terminateds

    def reward_function(self,dist,last_dist):
        # prepare rewrad function
        rf_name = f"rfv{args.rf_version}"
        rf_to_call = getattr(rfunctions, rf_name, None)
        assert callable(rf_to_call)

        rewards = {}
        p = self.player_now - 1
        _min_dist = self.min_dist[p]
        _max_total_r = self.max_total_r[p]
        _agent_total_r = self.agent_total_r[p]


        ##test
        # if self.player_now == 1:
        #     print(f"step {self.steps} player {self.player_now} action {act} distance {dist}")
        #     print(f'last_dist {self.last_dist}')
        ##
        if dist < _min_dist:
            # 当 dist 首次出现这么小, 那么计算后的 total reward 也应该比之前所有的都大
            factor = (1 + ( _min_dist - dist) / (_min_dist))
            if factor < 2:
                factor = 2
            if factor > 4:
                factor = 4
            r = (_max_total_r - _agent_total_r * args.gamma) * factor
            if r < 0.2:
                r = 0.2
            if r < 0:
                print(f'dist > _max_dist but r <0 {_max_total_r},{_agent_total_r},{dist},{_min_dist} ')
                r = 0.1
            self.max_total_r[p] = _agent_total_r * args.gamma + r

            # update min_dist
            self.min_dist[p] = dist

        else:
            r = rf_to_call(init_dist=self.init_dist[p], last_dist=last_dist, dist=dist, avg_dist=self.sw[p].current_avg)
        r = self.rs(r)[0]
        # update reward scaling

        for i in range(1, self.num_qubits + 1):
            if i == self.player_now:
                # update total reward for the current agent
                self.agent_total_r[i - 1] = self.agent_total_r[i - 1] * 0.99 + r
                # update max total r for the current agent
                if self.agent_total_r[i - 1] > self.max_total_r[i - 1]:
                    self.max_total_r[i - 1] = self.agent_total_r[i - 1]
                rewards.update({f'agent_{i}': r})
            # else:
            #     rewards.update({f'agent_{i}': 0})

        return rewards


    def distance_to_m(self,player) -> float:
        # 计算 player 到最近的 magic state 的距离
        x, y = self.chip.positions[player-1]
        dist = np.inf
        for mx,my in self.chip.magic_state:
            dist = min(dist, abs(x - mx) + abs(y - my))
        return dist

if __name__ == '__main__':
    #test code
    env = Env_3()
    env.reset()
    env.chip.print_state()
    player = 1
    for i in range(10):

        action = {f'agent_{player}': env.action_spaces[f'agent_{player}'].sample()}
        obs, rewards, terminateds, truncated, infos = env.step(action)
        player = ((player) % 4) + 1
        print(i)
        env.chip.print_state()
    env.chip.print_state()




