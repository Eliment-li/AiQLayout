import math
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
from gymnasium.envs.registration import current_namespace
from gymnasium.spaces import  Discrete, Box
from ray.rllib.env.multi_agent_env import  MultiAgentEnv
from shared_memory_dict import SharedMemoryDict

from core.agents import AgentsManager
from core.chip import Chip, QubitState, ChipLayoutType
from core.layout import ChipLayout, get_layout
from core.reward_function import RewardFunction
from core.reward_scaling import RewardScaling
from core.routing import a_star_path
from utils.calc_util import SlideWindow, normalize_MinMaxScaler
from utils.circuit_util import resize_2d_matrix, resize_3d_array
from utils.file.file_util import get_root_dir
from utils.ls_instructions import get_heat_map
from utils.position import positionalencoding2d
from utils.route_util import bfs_route

import config
args = config.RedisConfig()
rfunctions = RewardFunction()
'''
use agent manager to manage agents
actor act one by one
we compute the reward each round 
'''


def min_max_normalize(observation, min_value, max_value):
    """
    对 observation 进行 min-max 归一化。

    参数:
    - observation: numpy 数组，任意形状
    - min_value: 标量或与 observation 同形状的数组，全局最小值
    - max_value: 标量或与 observation 同形状的数组，全局最大值

    返回:
    - 归一化后的 observation，值在 [0, 1] 之间
    """
    # 防止分母为0
    denom = max_value - min_value
    denom = np.where(denom == 0, 1e-8, denom)
    normalized = (observation - min_value) / denom
    return normalized


rootdir = Path(get_root_dir())
class Env_6(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()
        self.num_qubits = args.num_qubits
        self.OBS_ROW = args.chip_rows
        self.OBS_COL = args.chip_cols


        print(f'OBS_ROW: {self.OBS_ROW}, OBS_COL: {self.OBS_COL}')

        self.lsi_file_path = rootdir / Path(args.lsi_file_path)
        print(self.lsi_file_path)

        self.heat_map = get_heat_map(file_path = self.lsi_file_path)
        self.RESIZE_HEATMAP = resize_2d_matrix(deepcopy(self.heat_map), r  = self.OBS_ROW,c = self.OBS_COL)
        self.RESIZE_HEATMAP = normalize_MinMaxScaler(self.RESIZE_HEATMAP)
        self.RESIZE_HEATMAP = [self.RESIZE_HEATMAP]

        print(f'init env_5 with {self.num_qubits} qubits')
        self.max_step = args.env_max_step * self.num_qubits

        chip_layout = ChipLayout(args.chip_rows, args.chip_cols, ChipLayoutType.EMPTY, self.num_qubits)
        self.chip = Chip(rows=args.chip_rows, cols=args.chip_cols,num_qubits=self.num_qubits,layout_type=ChipLayoutType.EMPTY,chip_layout=chip_layout)
        #agnet manager
        self.am = AgentsManager(self.num_qubits, self.chip)

        self.agents = self.possible_agents = [f"agent_{i+1}" for i in range(self.num_qubits)]
        self.a_space = Discrete(args.chip_rows * args.chip_cols)
        self.o_space =Box(
                            low=-5,
                            high=self.num_qubits + 1,
                            shape=(4+1+1+1,self.OBS_ROW,self.OBS_COL),
                            dtype=np.float32,
                            )


        self.observation_spaces = {f"agent_{i+1}": self.o_space for i in range(self.num_qubits)}
        self.action_spaces = {f"agent_{i+1}":  self.a_space for i in range(self.num_qubits)}
        self.pe = positionalencoding2d(self.chip.rows,self.chip.cols,4)
        self.sw = SlideWindow(50)
        self.r_scale = RewardScaling(shape=1, gamma=0.9)

        self.smd = SharedMemoryDict(name='env', size=10240)
        self.smd['min_dist'] = math.inf

        #self.gates = get_random_gates(num_qubits=self.num_qubits, size=args.gates_size)
        self.gates = []
        self.init_dist = self.get_init_dist()
        self.smd['init_dist'] = self.init_dist
        print(f'init_dist: {self.init_dist}')

    def get_init_dist(self):
        layout_type = ChipLayoutType(args.layout_type)
        layout = get_layout(layout_type=layout_type, rows=args.chip_rows, cols=args.chip_cols, num_qubits=self.num_qubits)
        temp_chip = Chip(rows=args.chip_rows, cols=args.chip_cols, num_qubits=self.num_qubits,
                         layout_type=layout.layout_type, chip_layout=layout)
        return self.compute_dist(temp_chip, self.am.activate_agent)[0]

    def reset(self, *, seed=None, options=None):
        self.steps = 1
        #self.chip.reset(layout_type=None)
        #may need clean qubits?
        self.am.reset_agents()
        self.dist_rec = [[] for i in range(self.num_qubits)]

        self.round_reward = 0
        self._max_total_r = -np.inf
        self._agent_total_r = 0
        self.round_last_dist = self.init_dist
        self.last_dist = self.init_dist
        self.min_sum_dist = self.init_dist

        # self.sw =SlideWindow(50)

        infos = {f'agent_{i + 1}':  'default' for i in range(self.num_qubits)}
        #infos = {f'agent_{i + 1}':  self.am(1).init_dist for i in range(self.num_qubits)}
        return self._get_obs(),infos

    def _get_obs(self):
        chip_state = deepcopy(self.chip.state)
        chip_state = min_max_normalize(chip_state,min_value=-5,max_value=self.num_qubits + 1)
        repeat_state = np.repeat(chip_state[np.newaxis, :, :], 4, axis=0)
        obs = repeat_state + self.pe
        pm =np.expand_dims(self.chip.position_mask(self.am.activate_agent), axis=0)
        obs = np.concatenate((obs,pm),axis = 0)  # (4, rows, cols) -> (4+1, rows, cols)

        chip_state = np.expand_dims(chip_state, axis=0)  # (rows, cols) -> (1, rows, cols)
        obs = np.concatenate((obs,chip_state),axis = 0) # (5, rows, cols) -> (4+1+1, rows, cols)

        zoom_factor = (self.OBS_ROW/self.chip.rows, self.OBS_COL/self.chip.cols)
        #resize
        #obs  = resize_3d_array(obs,zoom_factor)

        obs = np.concatenate((obs, self.RESIZE_HEATMAP), axis=0)  # (6, rows, cols) -> (4+1+1+1, rows, cols)
        # return {
        #     f'agent_{self.am.activate_agent}': obs
        # }

        # m = [1,4,7,19,20,49,50,79,80,92,95,98]
        # for index in m:
        #     if self.chip.valid_positions[index]!=0:
        #         print(self.chip.valid_positions)
        #         raise ValueError(f'valid position {index} is not 0')


        ret = {
            f'agent_{self.am.activate_agent}':{
                'observations': obs,
                'action_mask': deepcopy(self.chip.valid_positions)
            }
        }
        return ret

    def step(self, action):
        terminateds = self.is_terminated()
        act = action[f'agent_{self.am.activate_agent}']


        row = act // self.chip.cols
        col = act % self.chip.cols
        success = self.chip.goto(player=self.am.activate_agent, new_r=row, new_c=col)
        if not success:
            print(self.chip)
            print(self.chip.valid_positions)
            #raise ValueError(f'agent {self.am.activate_agent} move to ({row},{col}) failed at step {self.steps}')
            print(f'Warning! agent {self.am.activate_agent} move to ({row},{col}) failed at step {self.steps}, action: {act}, valid positions: {self.chip.valid_positions}')
        #for each step
        if self.steps >= self.num_qubits:
            dist, other_dist, self_dist = self.compute_dist(self.chip, self.am.activate_agent)
            if dist is  None:
                current_reward = self.r_scale(-4)
            else:
                self.sw.next(dist)
                self.last_dist = dist
                current_reward = self.reward_function(dist=dist, last_dist=self.last_dist)
        else:
            current_reward = 0
            dist = None
        #for each round
        if self.am.activate_agent == self.num_qubits:
            try:
                self.dist_rec[self.am.activate_agent - 1] = f'{dist}'
                # calc reward
                if dist is None:
                    self.round_reward = self.r_scale(-4)
                else:
                    self.round_reward = self.reward_function(dist=dist, last_dist=self.round_last_dist)
                    # self.sw.next(dist)
                    self.round_last_dist = dist
                # if not terminateds["__all__"]:
                #     self.chip.clean_qubits()
            except Exception as e:
                print(f'compute dist error: {e} at step {self.steps}')
                traceback.print_exc()

                self.chip.print_state()

        rewards = {f'agent_{self.am.activate_agent}': self.round_reward*0.2+ current_reward*0.8}

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

    def compute_dist_v2(self,chip:Chip, player:int):
        sum_dist= 0
        for i in range(self.num_qubits):
            j=0
            while j<i:
                cnt = self.heat_map[i][j]
                if cnt <= 0.000001:
                    j += 1
                    continue
                start = i + 1
                goal = j + 1
                sr, sc = chip.q_coor(start)
                gr, gc = chip.q_coor(goal)
                if j == QubitState.MAGIC.value:
                    path,dist,target_m= bfs_route(self.chip.state, start_row=sr, start_col=sc, target_values={QubitState.MAGIC.value})
                else:
                    path = a_star_path((sr, sc), (gr, gc), chip.state, goal)
                    dist = len(path)
                    if dist == 0:
                        return None, None, None
                j+=1
                sum_dist+=(cnt*dist)

        return sum_dist, None,None

    def compute_dist(self,chip:Chip, player:int):
        return self.compute_dist_v2(chip, player)

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

        reward_compen = False
        # p = self.am.activate_agent - 1
        # _min_dist = self.min_dist[p]
        if dist == None:
            #fail
            r = -2
        else:
            if (dist < self.min_sum_dist) and reward_compen:
                # 当 dist 首次出现这么小, 那么计算后的 total reward 也应该比之前所有的都大
                factor = 1.1
                r = (self._max_total_r - self._agent_total_r * args.gamma) * factor
                if r < 0.2:
                    r = 0.2
                if r < 0:
                    r = 0.1
            else:
                r = rf_to_call(init_dist=self.init_dist, last_dist=last_dist, dist=dist, avg_dist=self.sw.current_avg)
            #update _agent_total_r
            self._agent_total_r = self._agent_total_r * args.gamma + r
            # update _max_total_r for the current agent
            if self._agent_total_r > self._max_total_r:
                self._max_total_r = self._agent_total_r

        if (dist < self.min_sum_dist) and (dist < self.smd['min_dist']):
                # update min_dist
                self.min_sum_dist = dist
                self.smd['min_dist'] = dist
                self.smd['best_state'] = deepcopy(self.chip.state)

        if args.reward_scaling:
            r =self.r_scale(r)
        return r


if __name__ == '__main__':
    env = Env_6()
    env.reset()
    action = {f'agent_1':10}
    env.step(action)

    print(env.chip.state)
    print(env.chip.q_pos)
    print(env.chip.valid_positions)



