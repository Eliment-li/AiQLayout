#representing a quantum computing device chip
import math
import random
from copy import deepcopy
from enum import Enum

import numpy as np
import torch
from openpyxl.compat import deprecated
from torch import layout

from config import ConfigSingleton

from core.routing import bfs_find_target
from utils.position import positionalencoding2d
from utils.route_util import bfs_route

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

class QubitLayoutType(Enum):
    EMPTY = 0 #do nothing,do not put any extra qubit on the chip,
    GRID = 1
    GIVEN = 2
    RANDOM = 3


class Chip():

    def __init__(self,rows: int,cols: int, q_pos = [],layout_type = QubitLayoutType.EMPTY,num_qubits=None, broken_pos:list = []):
        '''
        :param rows:
        :param cols:
        :param broken: the broken postion of chip
        '''
        self.channel = 1
        self._cols = cols
        self._rows = rows
        self.q_pos = q_pos
        self._num_qubits = num_qubits
        '''
          must reset() before use 
        '''
        self.reset(q_pos = q_pos,layout_type=layout_type)

    def init_by_state(self, s: np.ndarray):
        self.state = deepcopy(s)
        #TODO init self._qubits_channel[i][j] = int
        #               self._position_mask[qubit - 1][i][j] = 1
        #               self.q_pos.append((i, j))



    def _init_qubits_layout(self,layout_type:QubitLayoutType,q_pos):
        if layout_type == QubitLayoutType.GRID:
            qubit = 1
            i = 1
            while i < self.rows - 1:
                j = 1
                while j< self.cols - 1:
                    if qubit > self._num_qubits:
                        return
                    if self.state[i][j] == 0 and self._broken_channel[i][j] == 0:
                        self.state[i][j] = qubit
                        self._qubits_channel[i][j] = qubit
                        self._position_mask[qubit - 1][i][j] = 1
                        self.q_pos.append((i, j))
                        qubit += 1
                    j+=2
                i+=2
            if qubit < self._num_qubits:
                print(f"Warning: only {qubit-1} qubits are placed, but {self._num_qubits} are required.")
                #TODO try random laytout for rest qubit
            else:
                return

        if len(q_pos) == 0:
            self.q_pos = [(None, None)] * self.num_qubits
            print('q_pos in reset is empty')
        else:
            i = 1
            for r, c in q_pos:
                if self.state[r][c] == 0:
                    self.state[r][c] = i
                    self._qubits_channel[r][c] = i
                    self._position_mask[i - 1][r][c] = 1
                    self.q_pos.append((r, c))
                    i += 1
                else:
                    continue
            # assert len(q_pos) == self.num_qubits, \
            #     f"len(q_pos) = {len(q_pos)} but self.num_qubits = {self.num_qubits} They should be equal"

    def clean_qubits(self):
        self.q_pos = [(None,None)]*self.num_qubits
        self._position_mask = np.zeros((self._num_qubits, self._rows, self._cols), dtype=np.float32)
        self._qubits_channel = np.zeros((self._rows, self._cols), dtype=np.float32)

        # if value >0 in self.state,make it to 0
        self.state[self.state > 0] = 0
        self.valid_positions = torch.ones((self._rows * self._cols))


    def reset(self,q_pos,layout_type: QubitLayoutType = QubitLayoutType.EMPTY):
       self.state = np.zeros((self._rows, self._cols), dtype=np.float32)
       self._broken_channel = np.zeros((self._rows, self._cols), dtype=np.float32)
       self._qubits_channel = np.zeros((self._rows, self._cols), dtype=np.float32)

       self._position_mask = np.zeros((self._num_qubits, self._rows, self._cols), dtype=np.float32)
       self._init_magic_state()
       # if args.enable_broken_patch:
       #     self._add_broken_patch()

       self._init_qubits_layout(q_pos=q_pos,layout_type=layout_type)
       self.valid_positions = torch.ones((self._rows * self._cols), dtype=torch.float32)
       #return the  flatten view of the state
       flatten_obs = self.state.ravel()
       for i in range(len(flatten_obs)):
           if flatten_obs[i] != 0:
               # 0= invalid 1=valid
               self.valid_positions[i] = 0

    def _add_broken_patch(self):
        broken = [(5,5),(5,6),(5,7),(5,8),(6,5),(7,5)]
        for x, y in broken:
            #self.state[x][y] = QubitState.BROKEN.value
            self.state[x][y] = QubitState.BROKEN.value
            self._broken_channel[x][y] = QubitState.BROKEN.value
            #TODO action mask



    def _random_init_qubits_layout(self):
        # vaild value start from _position[1] , -1 only for occupy
        i = 1
        while i <= self._num_qubits:
            x = random.randint(0, self._rows - 1)
            y = random.randint(0, self._cols - 1)
            if self.state[x][y] == 0 and self._broken_channel[x][y]== 0:

                self.state[x][y] = i
                self._qubits_channel[x][y] = i
                self._position_mask[i-1][x][y] = 1

                self.q_pos.append((x, y))
                i += 1
            else:
                continue

    '''
    从0行0列开始，逆时针旋转，转一圈回到起点，
    从第0个位置开始，每间隔两个空白，第三个元素置为 magic state
    '''
    def _init_magic_state(self):
        matrix = self.state
        rows = len(matrix)
        cols = len(matrix[0])

        # 定义四个方向的边界
        top = 0
        bottom = rows - 1
        left = 0
        right = cols - 1

        # 收集外围元素的顺序
        elements = []

        # 左列，从上到下
        for i in range(top, bottom + 1):
            elements.append((i, left))

        # 底行，从左到右（不包括第一个，因为左列已经包含）
        for j in range(left + 1, right + 1):
            elements.append((bottom, j))

        # 右列，从下到上（如果有多于一行）
        if top < bottom:
            for i in range(bottom - 1, top, -1):
                elements.append((i, right))

        # 顶行，从右到左（如果有多于一列）
        if left < right:
            for j in range(right, left, -1):
                elements.append((top, j))

        # 每隔两个元素将第三个元素设为-1
        count = 0
        for i, j in elements:
            count += 1
            if count % 3 == 0:
                matrix[i][j] = QubitState.MAGIC.value

        #return matrix



    def goto(self,player:int, new_r,new_c):
        if self.state[new_r, new_c] != 0:
            return False
        else:
            old_r, old_c = self.q_pos[player - 1]
            if old_r is not None and old_c is not None:
                self.state[old_r, old_c] = 0
                self._position_mask[player - 1][old_r, old_c] = 0
                self._qubits_channel[old_r, old_c] = 0
                self.valid_positions[old_r * self._rows + old_c ] = 1

            self.state[new_r, new_c] = player
            self._position_mask[player - 1][new_r, new_c] = 1
            self._qubits_channel[new_r, new_c] = player
            self.valid_positions[new_r * self._rows + new_c] = 0

            # occupy the new position
            self.q_pos[player - 1] = (new_r, new_c)
        # if torch.sum(self.valid_positions == 0).item() != (self.num_qubits+len(self._magic_state)):
        #     self.print_state()
        #     raise ValueError(f'number of 0(invalid position) '
        #                      f'should be {self.num_qubits+len(self._magic_state)}, '
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
    chip = Chip(15,15,layout_type = QubitLayoutType.GRID,num_qubits=50)
    chip.print_state()
    # chip.reset()
    print(chip.valid_positions)

    # for i in range(10000):
    #     player = random.randint(1,10)
    #     x = random.randint(0,9)
    #     y = random.randint(0,9)
    #     chip.goto(player,x,y)
    # chip.print_state()
    # print(chip.valid_positions)



