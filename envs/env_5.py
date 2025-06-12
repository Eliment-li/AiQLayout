import math
from copy import deepcopy
from pathlib import Path
from pprint import pprint
import  random

import gymnasium as gym
import numpy as np
from gymnasium import register
from gymnasium.spaces import  Discrete, Box,Dict
from numpy import dtype
from ray.rllib.env.multi_agent_env import  MultiAgentEnv

from config import ConfigSingleton
from core.agents import AgentsManager
from core.chip import Chip, ChipAction
from core.reward_function import RewardFunction
from core.reward_scaling import RewardScaling
from core.routing import a_star_path
from utils.calc_util import SlideWindow
from utils.circuit_util import get_gates_fixed
from utils.csv_util import append_data
from utils.position import positionalencoding2d

args = ConfigSingleton().get_args()
rfunctions = RewardFunction()
'''
use agent manager to manage agents
actor act one by one
we compute the reward each round 
'''
init_q_pos  = [
            (0,1),
            (0,2),
            (0,3),
            (0,4),
            (0,5),

            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
]
class Env_5(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()
        self.steps = 0
        self.num_qubits = args.num_qubits
        print(f'init env_5 with {self.num_qubits} qubits')
        self.max_step = args.env_max_step * self.num_qubits
        # define chip
        self.DoneAct = args.chip_rows * args.chip_cols
        self.chip = Chip(rows=args.chip_rows, cols=args.chip_cols,num_qubits=self.num_qubits,q_pos=[])
        self.chip.print_state()
        #agnet manager
        self.am = AgentsManager(self.num_qubits, self.chip)

        self.agents = self.possible_agents = [f"agent_{i+1}" for i in range(self.num_qubits)]

        self.a_space = Discrete(args.chip_rows * args.chip_cols)
        self.o_space =Box(
                            low=-5,
                            high=self.num_qubits + 1,
                            shape=(4+1+1,args.chip_rows,args.chip_cols),
                            dtype=np.float32,
                            )

        # self.o_space = Dict(
        #     {
        #         "action_mask": Box(0.0, 1.0, shape=(self.a_space.n,)),
        #         "observations":Box(
        #                     low=-5,
        #                     high=self.num_qubits + 1,
        #                     shape=(4+1,args.chip_rows,args.chip_cols),
        #                     dtype=np.float32,
        #                     )
        #
        #
        #     }
        # )
        self.observation_spaces = {f"agent_{i+1}": self.o_space for i in range(self.num_qubits)}
        self.action_spaces = {f"agent_{i+1}":  self.a_space for i in range(self.num_qubits)}
        self.pe = positionalencoding2d(self.chip.rows,self.chip.cols,4)
        self.sw = SlideWindow(50)

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.chip.reset(q_pos=[])
        self.chip.clean_qubits()
        self.am.reset_agents()
        self.dist_rec = [[] for i in range(self.num_qubits)]

        temp_chip = Chip(rows=args.chip_rows, cols=args.chip_cols,num_qubits=self.num_qubits,q_pos=init_q_pos)
        self.min_sum_dist = self.compute_dist(temp_chip,self.am.activate_agent)[0]

        #clean the init position

        self.reward = 0
        self._max_total_r = -np.inf
        self._agent_total_r = 0
        self.last_dist = self.min_sum_dist
        self.init_dist = self.min_sum_dist
        # self.sw =SlideWindow(50)

        infos = {f'agent_{i + 1}':  'default' for i in range(self.num_qubits)}
        #infos = {f'agent_{i + 1}':  self.am(1).init_dist for i in range(self.num_qubits)}
        return self._get_obs(),infos

    def _get_obs(self):
        repeat_state = np.repeat(self.chip.state[np.newaxis, :, :], 4, axis=0)
        obs = repeat_state + self.pe
        pm =np.expand_dims(self.chip.position_mask(self.am.activate_agent), axis=0)
        obs = np.concatenate((obs,pm),axis = 0)  # (4, rows, cols) -> (4+1, rows, cols)
        chip_state = deepcopy(self.chip.state)
        chip_state = np.expand_dims(chip_state, axis=0)  # (rows, cols) -> (1, rows, cols)
        obs = np.concatenate((obs,chip_state),axis = 0) # (5, rows, cols) -> (4+1+1, rows, cols)
        # return {
        #     f'agent_{self.am.activate_agent}': obs
        # }
        ret = {
            f'agent_{self.am.activate_agent}':{
                'observations': obs,
                'action_mask': self.chip.valid_positions
            }
        }
        return ret

    def step(self, action):
        terminateds = self.is_terminated()
        act = action[f'agent_{self.am.activate_agent}']
        rewards = {f'agent_{self.am.activate_agent}': self.reward}
        # if act == self.DoneAct or self.am.is_done(self.am.activate_agent):
        #     self.am.set_done(self.am.activate_agent)
            #dist, other_dist, self_dist = self.compute_dist(self.am.activate_agent)
        # else:
        row = act // self.chip.cols
        col = act % self.chip.cols
        success = self.chip.goto(player=self.am.activate_agent, new_r=row, new_c=col)
        assert success, f'agent {self.am.activate_agent} move to ({row},{col}) failed at step {self.steps}'
        if self.am.activate_agent == self.num_qubits:
            ##TODO  clean qubits?
            #clean qubits
            #self.chip.clean_qubits()
            try:
                dist, other_dist, self_dist = self.compute_dist(self.chip,self.am.activate_agent)
                self.dist_rec[self.am.activate_agent - 1] = f'{dist}'
                # calc reward
                if dist is None:
                    terminateds = {"__all__": True}
                    rewards = {f'agent_{self.am.activate_agent}': -2}
                else:
                    self.sw.next(dist)
                    self.reward = self.reward_function(dist=dist, last_dist=self.last_dist)
                    self.last_dist = dist
            except Exception as e:
                print(f'compute dist error: {e} at step {self.steps}')
                self.chip.print_state()


        truncated = {}
        infos = self._get_infos()
        #switch to next agent
        self.am.switch_next()
        self.steps += 1
        return self._get_obs(),rewards,terminateds,truncated,infos
    def _get_infos(self):
        return {
                    f'agent_{self.am.activate_agent}':
                    {
                        'distance': self.dist_rec[self.am.activate_agent - 1],
                        'max_total_r':self._max_total_r
                    }
                 }

    def compute_dist(self,chip:Chip, player:int):
        gates = get_gates_fixed()

        depth = 1
        new = True
        layer = deepcopy(chip.state)

        i = 0
        other_dist = 0
        self_dist = 0
        while i < len(gates):
            start, goal = gates[i]

            sr,sc = chip.q_coor(start)
            gr,gc =  chip.q_coor(goal)
            path = a_star_path( (sr,sc), ( gr,gc), layer,goal)

            if start == player or goal == player:
                self_dist += len(path)
            else:
                other_dist += len(path)

            if len(path) == 2:
                #the two qubits are already connected
                i += 1
                continue
            elif len(path)==0:
                if new:
                    #已经刷新过但是无法找到路径
                    # print('path = 0')
                    # print(f'agent {player} the dist is None in compute_dist')
                    # print(f'from{start}({sr},{sc}) to {goal} gr,gc=({gr},{gc})')
                    # print(self.chip.q_pos)
                    # print(layer)
                    # path = Path(args.results_evaluate_path, (args.time_id + '_results.csv'))
                    # append_data(file_path=path,data=str(self.chip.state))
                    return None,None,None
                else:
                    layer = deepcopy(chip.state)
                    depth += 1
                    new = True
            else:
                #occupy the path
                for p in path:
                    layer[p[0]][p[1]] = -3
                new = False
                i+=1

        sum = other_dist + self_dist
        return sum,other_dist,self_dist

    def is_terminated(self):

        if self.steps >= self.max_step:
           terminateds = {"__all__": True}
        elif self.am.is_all_done():
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

        p = self.am.activate_agent - 1
        # _min_dist = self.min_dist[p]
        if dist == None:
            #fail
            r = -2
        elif (dist < self.min_sum_dist ):
            # 当 dist 首次出现这么小, 那么计算后的 total reward 也应该比之前所有的都大
            # factor = (1 + ( _min_dist - dist) / (_min_dist))
            factor = 1.1
            r = (self._max_total_r - self._agent_total_r * args.gamma) * factor
            if r < 0.2:
                r = 0.2
            if r < 0:
                r = 0.1
            self._agent_total_ = self._agent_total_r * args.gamma + r

            # update min_dist
            self.min_sum_dist = dist

        else:
            r = rf_to_call(init_dist=self.init_dist, last_dist=last_dist, dist=dist, avg_dist=self.sw.current_avg)

            self._agent_total_r = self._agent_total_r * args.gamma + r
            # update max total r for the current agent
            if self._agent_total_r > self._max_total_r:
                self._max_total_r = self._agent_total_r
        return r


if __name__ == '__main__':
    pass


