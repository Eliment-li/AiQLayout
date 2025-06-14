#representing a quantum computing device chip
import math
import random
from copy import deepcopy
from enum import Enum

import numpy as np
import torch
from openpyxl.compat import deprecated

from config import ConfigSingleton

from core.routing import bfs_find_target
from utils.position import positionalencoding2d

args = ConfigSingleton().get_args()

class ChipAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4
    Done = 5

class QubitState(Enum):
    ANCILLA = 0  # free to use
    MAGIC = -1 # magic state
    BROKEN = -2 #broken, unavailable



class Chip():
    def __init__(self,rows: int,cols: int, q_pos = None,num_qubits=None, broken_pos:list = []):
        '''
        :param rows:
        :param cols:
        :param broken: the broken postion of chip
        '''
        self.channel = 1
        self._cols = cols
        self._rows = rows
        if num_qubits:
            self._num_qubits = num_qubits
        else:
            self._num_qubits = args.num_qubits
        self.reset(q_pos = q_pos)


    def _random_init_qubits_layout(self):
        # vaild value start from _position[1] , -1 only for occupy
        i = 1
        while i <= self._num_qubits:
            x = random.randint(0, self._rows - 1)
            y = random.randint(0, self._cols - 1)
            if self._state[x][y] == 0 and self._broken_channel[x][y]== 0:

                self._state[x][y] = i
                self._qubits_channel[x][y] = i
                self._position_mask[i-1][x][y] = 1

                self._q_pos.append((x, y))
                i += 1
            else:
                continue

    def _init_qubits_layout(self,q_pos):

        assert len(q_pos) == self.num_qubits, \
            f"len(q_pos) = {len(q_pos)} but self.num_qubits = {self.num_qubits} They should be equal"
        i = 1
        for r, c in q_pos:
            if self._state[r][c] == 0:

                self._state[r][c] = i
                self._qubits_channel[r][c] = i
                self._position_mask[i - 1][r][c] = 1
                self._q_pos.append((r, c))
                i += 1
            else:
                continue

    def clean_qubits(self):
        self._q_pos = [(None,None)]*self.num_qubits
        self._position_mask = np.zeros((self._num_qubits, self._rows, self._cols), dtype=np.float32)
        self._qubits_channel = np.zeros((self._rows, self._cols), dtype=np.float32)

        # if value >0 in self._state,make it to 0
        self._state[self._state > 0] = 0
        self.valid_positions = torch.ones((self._rows * self._cols))


    def reset(self,q_pos = None):
       self._state = np.zeros((self._rows, self._cols), dtype=np.float32)
       self._broken_channel = np.zeros((self._rows, self._cols), dtype=np.float32)
       self._qubits_channel = np.zeros((self._rows, self._cols), dtype=np.float32)

       self._position_mask = np.zeros((self._num_qubits, self._rows, self._cols), dtype=np.float32)
       # 0= valid 1=invalid
       self.valid_positions = torch.ones((self._rows * self._cols),
                                         dtype=torch.float32)  # np.zeros((self._rows * self._cols), dtype=float)
       #self._init_magic_state()
       if args.enable_broken_patch:
           self._add_broken_patch()

       if q_pos is None:
            print('q_pos in reset is None')
            self._q_pos = []
            self. _random_init_qubits_layout()
       elif len(q_pos) ==0 :
            self._q_pos = [(None, None)] * self.num_qubits
            print('q_pos in reset is empty')
       else:
           #qubits must be init in the last
           self._q_pos = []
           self._init_qubits_layout(q_pos)


       for r, c in self._q_pos:
           if r is not None and c is not None :
              self.valid_positions[r * self._rows + c] = 0

    def _add_broken_patch(self):
        broken = [
            (5,5),
            (5,6),
            (5,7),
            (5,8),

            (6,5),
            (7,5),

        ]
        for x, y in broken:
            #self._state[x][y] = QubitState.BROKEN.value
            self._state[x][y] = QubitState.BROKEN.value
            self._broken_channel[x][y] = QubitState.BROKEN.value


    def _init_magic_state(self):
        self._magic_state = [
            # (0, 0),
            # (0, self._cols - 1),
            # (self._rows - 1, 0),
            (self._rows - 1, self._cols - 1),
        ]
        for x, y in self._magic_state:
            self._state[x][y] = QubitState.MAGIC.value

    def goto(self,player:int, new_r,new_c):
        if self._state[new_r, new_c] != 0:
            return False
        else:
            old_r, old_c = self._q_pos[player - 1]
            if old_r is not None and old_c is not None:
                self._state[old_r, old_c] = 0
                self._position_mask[player - 1][old_r, old_c] = 0
                self._qubits_channel[old_r, old_c] = 0

                self.valid_positions[old_r * self._rows + old_c ] = 1

            self._state[new_r, new_c] = player
            self._position_mask[player - 1][new_r, new_c] = 1
            self._qubits_channel[new_r, new_c] = player
            self.valid_positions[new_r * self._rows + new_c] = 0

            # occupy the new position
            self._q_pos[player - 1] = (new_r, new_c)
        if torch.sum(self.valid_positions == 0).item() > self.num_qubits:
            print('number of 0 too many')
            self.print_state()
        return True


    def move(self, player: int, act:int):
        old_r,old_c = self._q_pos[player - 1]

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
                or self._state[new_r,new_c] != 0):

            return False
        else:
            #free the old position
            self._state[old_r, old_c] = 0
            self._state[new_r, new_c] = player

            self._position_mask[player - 1][old_r, old_c] = 0
            self._position_mask[player - 1][new_r, new_c] = 1

            self._qubits_channel[old_r, old_c] = 0
            self._qubits_channel[new_r, new_c] = player

            #occupy the new position
            self._q_pos[player - 1] = (new_r, new_c)
            return True

    def route_to_magic_state(self, player: int):
        '''
        :param player:
        :return: the length to magic state
        '''
        px,py = self._q_pos[player - 1]
        # use dfs to find the shortest path to magic state(value that equal to -1)

        path_len,path = bfs_find_target(self._state, px, py)
        return path_len

    def __str__(self):
        # 设置每个元素的宽度
        element_width = 2
        result = []  # 用于存储每一行的字符串
        for row in self._state:
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
            conflict_q_pos = list(zip(conflicts[0], conflicts[1]))
            raise ValueError(f"在位置 {conflict_q_pos} 处发现非零元素冲突 {arr1},\n {arr2}")

        # 合并数组，非零元素优先
        result = np.where(arr1 != 0, arr1, arr2)
        return result

    def position_mask(self,player):
        return self._position_mask[player - 1]

    def q_coor(self,player):
        return self._q_pos[player - 1]




    @property
    # start from index 1
    def q_pos(self):
        return self._q_pos

    @property
    def state(self):
        return  self._state
    @property
    def channel_state(self):
        s = np.array([self._qubits_channel,self._broken_channel]).astype(np.int16)
        return s


    def print_state(self):
        # 设置每个元素的宽度
        element_width = 2
        for row in self._state:
            # 使用列表推导式将 0 替换为 '--'
            replaced_row = ['--' if value == 0 else int(value) for value in row]
            # :>{element_width} 指定右对齐，并确保每个值占用固定的宽度。
            # str(value) 将值转换为字符串
            formatted_row = [f"{str(value):>{element_width}}" for value in replaced_row]
            print(" ".join(formatted_row))

    @property
    def magic_state(self):
        return self._magic_state

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def num_qubits(self):
        return self._num_qubits

    def plot(self):
        pass

#test code
if __name__ == '__main__':
    q_pos = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    ]
    chip = Chip(6,6,q_pos=[],num_qubits=5)
    chip.reset()
    chip.print_state()
    print(chip.valid_positions)

    for i in range(100):
        player = random.randint(1,5)
        x = random.randint(0,5)
        y = random.randint(0,5)
        chip.goto(player,x,y)
    chip.print_state()
    print(chip.valid_positions)


