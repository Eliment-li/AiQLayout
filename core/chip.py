#representing a quantum computing device chip
import math
import random
from copy import deepcopy
from enum import Enum
from pathlib import Path

import numpy as np
import torch

import config


from core.layout import ChipLayoutType, ChipLayout, QubitState, get_layout

from core.routing import bfs_find_target, a_star_path
from utils.file.file_util import get_root_dir
from utils.ls_instructions import get_heat_map
from utils.position import positionalencoding2d
from utils.route_util import bfs_route

# args = config.get_args()

INVALID_POS= 0
VALID_POS= 1

class ChipAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4
    Done = 5


class Chip():

    def __init__(self,rows: int = None,cols: int= None ,num_qubits:int= None,layout:ChipLayout=None):
        self.channel = 1
        self.num_qubits = num_qubits or layout.num_qubits
        self._cols = cols or layout.cols
        self._rows = rows or layout.rows
        self.q_pos = [(None, None)] * self.num_qubits
        self.layout = layout
        self._qubits_channel = np.zeros((self._rows, self._cols), dtype=np.float32)
        self._position_mask = np.zeros((self.num_qubits, self._rows, self._cols), dtype=np.float32)

        '''
          must reset() before use 
        '''
        self.reset(self.layout.layout_type)


    def reset(self,layout_type:ChipLayoutType=None):

        self._init_qubits_layout(layout_type)
        # return the  flatten view of the state
        flatten_obs = self.state.ravel()
        self.valid_positions = torch.ones((self._rows * self._cols), dtype=torch.float32)
        for i in range(len(flatten_obs)):
            if flatten_obs[i] != 0:
                # 0= invalid 1=valid
                self.valid_positions[i] = INVALID_POS


    def set_state(self, s: np.ndarray):
        self.state = deepcopy(s)
        #TODO init self._qubits_channel[i][j] = int
        #               self._position_mask[qubit - 1][i][j] = 1
        #               self.q_pos.append((i, j))


    def _init_qubits_layout(self,layout_tpye):
        if layout_tpye is None:
            return
        self.state = deepcopy(self.layout.state)
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                qubit = self.state[i][j]
                if qubit > 0:
                    self._qubits_channel[i][j] = qubit
                    self._position_mask[qubit - 1][i][j] = 1
                    self.q_pos[qubit - 1] =(i, j)
            # assert len(q_pos) == self.num_qubits, \
            #     f"len(q_pos) = {len(q_pos)} but self.num_qubits = {self.num_qubits} They should be equal"

    # def clean_qubits(self):
    #     self.q_pos = [(None,None)]*self.num_qubits
    #     self._position_mask = np.zeros((self.num_qubits, self._rows, self._cols), dtype=np.float32)
    #     self._qubits_channel = np.zeros((self._rows, self._cols), dtype=np.float32)
    #
    #     # if value >0 in self.state,make it to 0
    #     self.state[self.state > 0] = 0
    #     self.valid_positions = torch.ones((self._rows * self._cols))


    def goto(self,player:int, new_r,new_c):
        if self.state[new_r, new_c] != 0:
            return False
        else:
            old_r, old_c = self.q_pos[player - 1]
            if old_r is not None and old_c is not None:
                self.state[old_r, old_c] = 0
                self._position_mask[player - 1][old_r, old_c] = 0
                self._qubits_channel[old_r, old_c] = 0
                self.valid_positions[old_r * self._rows + old_c ] = VALID_POS

            self.state[new_r, new_c] = player
            self._position_mask[player - 1][new_r, new_c] = 1
            self._qubits_channel[new_r, new_c] = player
            self.valid_positions[new_r * self._rows + new_c] = INVALID_POS

            # occupy the new position
            self.q_pos[player - 1] = (new_r, new_c)
        # if torch.sum(self.valid_positions == 0).item() != (self.num_qubits+len(self.magic_state)):
        #     self.print_state()
        #     raise ValueError(f'number of 0(invalid position) '
        #                      f'should be {self.num_qubits+len(self.magic_state)}, '
        #                      f'but now is{torch.sum(self.valid_positions == 0).item()}' )
        return True


    def move(self, player: int, act:int):
        old_r,old_c = self.q_pos[player - 1]

        assert act in ChipAction, f"{act} is not a valid action"

        match act:
            case  ChipAction.LEFT.value:
                new_c,new_r = old_c - 1, old_r  # left
            case ChipAction.RIGHT.value:
                new_c,new_r  = old_c + 1, old_r   # right
            case  ChipAction.UP.value:
                new_c,new_r  = old_c, old_r - 1 # up
            case ChipAction.DOWN.value:
                new_c,new_r  = old_c, old_r + 1  # down
            case ChipAction.STAY.value:
                return True
            case ChipAction.Done.value:
                return
            case _:
                pass

        #if new_post out of matrix
        if (
                new_c < 0
                or new_c >= self._cols
                or new_r < 0
                or new_r >= self._rows
                or self.state[new_r,new_c] != 0):

            return False
        else:
            #free the old position
            self.state[old_r, old_c] = 0
            self.state[new_r, new_c] = player

            self._position_mask[player - 1][old_r, old_c] = 0
            self._position_mask[player - 1][new_r, new_c] = 1

            self._qubits_channel[old_r, old_c] = 0
            self._qubits_channel[new_r, new_c] = player

            #occupy the new position
            self.q_pos[player - 1] = (new_r, new_c)
            return True

    def route_to_magic_state(self, player: int):
        '''
        :param player:
        :return: the length to magic state
        '''
        px,py = self.q_pos[player - 1]
        # use dfs to find the shortest path to magic state(value that equal to -1)

        path_len,path = bfs_find_target(self.state, px, py)
        return path_len

    def __str__(self):
        # 设置每个元素的宽度
        element_width = 2
        result = []  # 用于存储每一行的字符串
        for row in self.state:
            # 使用列表推导式将 0 替换为 '--'
            replaced_row = ['--' if value == 0 else int(value) for value in row]
            # :>{element_width} 指定右对齐，并确保每个值占用固定的宽度。
            # str(value) 将值转换为字符串
            formatted_row = [f"{str(value):>{element_width}}" for value in replaced_row]
            result.append(" ".join(formatted_row))
        return "\n".join(result)  # 将所有行拼接成一个字符串，用换行符分隔

    def merge_states(self,arr1, arr2):
        # 检查形状是否相同
        if arr1.shape != arr2.shape:
            raise ValueError("输入数组的形状必须相同")

        # 检查是否有重叠的非零元素
        overlap = (arr1 != 0) & (arr2 != 0)
        if np.any(overlap):
            # 找出冲突位置
            conflicts = np.where(overlap)
            conflictq_pos = list(zip(conflicts[0], conflicts[1]))
            raise ValueError(f"在位置 {conflictq_pos} 处发现非零元素冲突 {arr1},\n {arr2}")

        # 合并数组，非零元素优先
        result = np.where(arr1 != 0, arr1, arr2)
        return result

    def position_mask(self,player):
        return self._position_mask[player - 1]

    def q_coor(self,player):
        return self.q_pos[player - 1]



    @property
    def channel_state(self):
        s = np.array([self._qubits_channel,self._broken_channel]).astype(np.int16)
        return s


    def print_state(self):
        # 设置每个元素的宽度
        element_width = 2
        for row in self.state:
            # 使用列表推导式将 0 替换为 '--'
            replaced_row = ['--' if value == 0 else int(value) for value in row]
            # :>{element_width} 指定右对齐，并确保每个值占用固定的宽度。
            # str(value) 将值转换为字符串
            formatted_row = [f"{str(value):>{element_width}}" for value in replaced_row]
            print(" ".join(formatted_row))



    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols


    def plot(self):
        pass

# def test():
#     layout = get_layout(name=ChipLayoutType.COMPACT_1, rows=12, cols=12, num_qubits=20)
#     chip = Chip(12,12,layout_type = ChipLayoutType.GRID,num_qubits=20,layout=layout)
#     chip.print_state()
#     print(chip.q_pos)
#
#
#     for i in range(10000):
#         player = random.randint(1,20)
#         x = random.randint(0,9)
#         y = random.randint(0,9)
#         chip.goto(player,x,y)
#
#     chip.print_state()
#     print(chip.valid_positions)



def compute_dist_v2(num_qubits, heat_map,chip:Chip):
    sum_dist = 0
    for i in range(num_qubits):
        j = 0
        while j < i:
            cnt = heat_map[i][j]
            if cnt <=0.000001:
                j += 1
                continue
            start = i + 1
            goal = j + 1
            sr, sc = chip.q_coor(start)
            gr, gc = chip.q_coor(goal)
            if j == QubitState.MAGIC.value:
                path, dist, target_m = bfs_route(chip.state, start_row=sr, start_col=sc,
                                                 target_values={QubitState.MAGIC.value})
            else:
                path = a_star_path((sr, sc), (gr, gc), chip.state, goal)
                dist = len(path)

            if dist == 0:
                return None, None, None
            j += 1
            sum_dist += (cnt * dist)

    return sum_dist


def compute_depth(num_qubits, heat_map,chip:Chip):
    depth = 0
    new = True
    layer = deepcopy(chip.state)
    for i in range(num_qubits):
        j = 0
        while j < i:
            cnt = heat_map[i][j]
            if cnt <=0.000001:
                j += 1
                continue
            start = i + 1
            goal = j + 1
            sr, sc = chip.q_coor(start)
            gr, gc = chip.q_coor(goal)
            if j == QubitState.MAGIC.value:
                path, dist, target_m = bfs_route(layer, start_row=sr, start_col=sc,
                                                 target_values={QubitState.MAGIC.value})
            else:
                path = a_star_path((sr, sc), (gr, gc), layer, goal)
                dist = len(path)

            if dist == 0:
                if new:
                    return None
                else:
                    layer = deepcopy(chip.state)
                    depth += cnt
                    new = True
            else:
                for p in path:
                    layer[p[0]][p[1]] = -3
                new = False
                j += 1
    return depth

def benchmark_layouts(layout_type: ChipLayoutType = None,num_qubits: int = 0, size: int=0,heat_map=None):
    layout_type = ChipLayoutType(layout_type)
    layout = get_layout(layout_type=layout_type, rows=size, cols=size,num_qubits=num_qubits)
    # layout = ChipLayout(rows=args.chip_rows,cols=args.chip_cols,layout_type = ChipLayoutType.GRID,num_qubits=self.num_qubits)#get_layout(name = ChipLayoutType.GRID, rows=args.chip_rows, cols=args.chip_cols, num_qubits=self.num_qubits)
    temp_chip = Chip(rows=size, cols=size, num_qubits=num_qubits,layout=layout)
    #dist  = compute_dist_v2(num_qubits, heat_map, temp_chip)
    dist  = compute_depth(num_qubits, heat_map, temp_chip)
    return dist



#

#test code
if __name__ == '__main__':
    rootdir = get_root_dir()
    lsi_size = [10,15,20,25]

    LAYOUT = [
        ChipLayoutType.LINER_1,
        ChipLayoutType.COMPACT_1,
        ChipLayoutType.COMPACT_2
    ]
    chip_size = [
        17,12,18
    ]

    for j in range(len(chip_size)):
        layout = LAYOUT[j]
        size = chip_size[j]
        print(layout)
        for i in range(4):
            num_qubits = lsi_size[i]
            file = Path(f'assets/circuits/qft/LSI_qftentangled_indep_qiskit_{num_qubits}.lsi')
            heat_map = get_heat_map(file_path=rootdir / file)
            dist = benchmark_layouts(layout_type=ChipLayoutType(layout), num_qubits=num_qubits, size=size,heat_map=heat_map)
            print(dist)

    # state = np.array([[ 0,  0,  0, -1,  0,  0, -1,  0,  0, -1,  0,  0],
    #    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    #    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  0,  4,  7,  0,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  5,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    #    [-1,  0,  3,  1,  8,  0,  0,  0,  0,  0,  0,  0],
    #    [ 0, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  9,  0,  0,  0,  0,  0,  0,  0,  0, -1],
    #    [-1,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  0,  6,  0,  0,  0,  0,  0,  0,  0, -1],
    #    [-1,  0,  0, -1,  0,  0, -1,  0,  0, -1,  0,  0]])
    # size = 12
    # num_qubits = 10
    # file = Path(f'assets/circuits/qft/LSI_qftentangled_indep_qiskit_{num_qubits}.lsi')
    # heat_map = get_heat_map(file_path=rootdir / file)
    # layout = get_layout(layout_type=ChipLayoutType.GIVEN, rows=size, cols=size, num_qubits=num_qubits,given_state=state)
    # temp_chip = Chip(rows=size, cols=size, num_qubits=num_qubits, layout=layout)
    # dist = compute_depth(num_qubits, heat_map, temp_chip)
    # print(dist)



