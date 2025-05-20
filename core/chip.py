#representing a quantum computing device chip
import math
import random
from copy import deepcopy
from enum import Enum

import numpy as np
from config import ConfigSingleton

from core.routing import bfs_find_target

args = ConfigSingleton().get_args()

class ChipAction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4

class QubitState(Enum):
    ANCILLA = 0  # free to use
    MAGIC = -1 # magic state
    BROKEN = -2 #broken, unavailable



class Chip():
    def __init__(self,rows: int,cols: int, num_qubits=None, disable:list = []):
        '''
        :param rows:
        :param cols:
        :param disable: the broken postion of chip
        '''

        self._cols = cols
        self._rows = rows
        if num_qubits:
            self._num_qubits = num_qubits
        else:
            self._num_qubits = args.num_qubits
        #start from index 1
        self._positions=[]
        self._state=np.zeros(( self._rows,self._cols), dtype=np.int16)
        # magic state
        self._magic_state = []
        self._magic_state_pos=[]
        self._init_magic_state()
        self.random_init()
        self.last_distance=[]


    def get_positon(self,player):
        return self._positions[player - 1]
    def random_init(self):
        # vaild value start from _position[1] , -1 only for occupy
        i = 1
        while i <= self._num_qubits:
            x = random.randint(0, self._rows - 1)
            y = random.randint(0, self._cols - 1)
            if self._state[x][y] == 0:
                self._state[x][y] = i
                self._positions.append((x, y))
                i += 1
            else:
                continue

    def _init_qubits_layout(self):
       # init qubits one by one
        for r in range(self.rows):
            for c in range(self.cols):
                if self._state[r][c] == 0:
                    self._positions.append((r, c))
                    self._state[r][c] = len(self._positions)
                    if len(self._positions) == self._num_qubits:
                        return

    def reset(self):
       self._state = np.zeros((self._rows, self._cols), dtype=np.int32)
       self._positions = []
       self._init_magic_state()
       self.random_init()
       self._magic_state = []

    def _init_magic_state(self):
        self._magic_state = [
            (0, 0),
            (0, self._cols - 1),
            (self._rows - 1, 0),
            (self._rows - 1, self._cols - 1),
        ]
        for x, y in self._magic_state:
            self._state[x][y] = QubitState.MAGIC.value
            self._magic_state_pos.append((x, y))

    def move(self, player: int, act:int):
        old_r,old_c = self._positions[player - 1]

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
            case _:
                pass

        #print(f"old_x:{old_x}, old_y:{old_y}, new_c:{new_c}, new_r:{new_r}")
        #if new_post out of matrix
        if (
                new_c < 0
                or new_c >= self._cols
                or new_r < 0
                or new_r >= self._rows
                or self._state[new_r,new_c] != 0):

            # print("Invalid move")
            return False
        else:
            #print(f'player{player} move {act} ')
            #free the old position
            self._state[old_r, old_c] = 0
            #occupy the new position
            self._positions[player - 1] = (new_r, new_c)
            self._state[new_r, new_c] = player

            return True

    def route_to_magic_state(self, player: int):
        '''
        :param player:
        :return: the length to magic state
        '''
        px,py = self._positions[player - 1]
        # use dfs to find the shortest path to magic state(value that equal to -1)

        path_len,path = bfs_find_target(self._state, px, py)
        return path_len


    def __str__(self):
        return self._state.__str__()

    @property
    # start from index 1
    def position(self):
        return self._positions

    @property
    def magic_state_pos(self):
        return self._magic_state_pos

    @property
    def state(self):
        # state = deepcopy(self._state)
        # for i in range(1,self._num_qubits+1):
        #     x,y = self._positions[i]
        #     state[x][y] = i*100
        return  self._state

    def print_state(self):
        for row in self._state:
            # 使用列表推导式将0替换为'-'
            replaced_row = ['-' if value == 0 else value for value in row]
            # 打印替换后的行
            print(" ".join(map(str, replaced_row)))


    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def positions(self):
        return self._positions


    def plot(self):
        pass

#test code
if __name__ == '__main__':
    chip = Chip(5,5)
    print(chip)
    print(chip.position)
    chip.print_state()
    # for i in range(3):
    #     player = random.randint(1, chip._num_qubits)
    #     act  = random.randint(0, 3)
    #     chip.move(player, act)
    # print(chip.state)
    # print(chip.position)
    #
    # print('routing length  = ', chip.route_to_magic_state(1))
    # print('length  = ', chip.distance_to_others(4))

    # rewards = {}
    # for i in range(1, chip._num_qubits + 1):
    #     distance = chip.distance_to_others(i)
    #     print(distance)
    #     r = math.log(distance +1, 2)-3.5
    #     rewards.update({f'agent_{i}': r})
    #
    # print(rewards)
