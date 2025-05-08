#representing a quantum computing device chip
import math
import random
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

'''
TODO: use enum to warp action
'''
class Chip():
    def __init__(self,cols: int, rows: int, disable:list = []):
        '''
        :param rows:
        :param cols:
        :param disable: the broken postion of chip
        '''

        self._cols = cols
        self._rows = rows
        self._num_qubits = args.num_qubits
        #start from index 1
        self._positions=[]
        self._state=np.zeros((self.rows,self.cols), dtype=np.int32)
        # magic state
        self._magic_state = []
        self._init_magic_state()
        self._init_qubits_layout()
        self.last_distance=[]



    def random_init(self):
        # vaild value start from _position[1] , -1 only for occupy
        self._positions = [-1]
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
       # random init qubits layout
       # init qubits one by one
        self._positions = [-1]
        for r in range(self.rows):
            for c in range(self.cols):
                if self._state[r][c] == 0:
                    self._positions.append((r, c))
                    self._state[r][c] = len(self._positions)-1
                    if len(self._positions)-1 == self._num_qubits:
                        return

    def reset(self):
       self._state = np.zeros((self._rows, self._cols), dtype=np.int32)
       self._init_magic_state()
       self._init_qubits_layout()

    def _init_magic_state(self):
        pass
        # self._magic_state = [
        #     (0, 0),
        #     (0, self._cols - 1),
        #     (self._rows - 1, 0),
        #     (self._rows - 1, self._cols - 1),
        # ]
        #
        # for x, y in self._magic_state:
        #     self._state[x][y] = -1

    def move(self, player: int, act:int):
        old_x,old_y = self._positions[player]

        assert act in ChipAction, f"{act} is not a valid action"

        if act == ChipAction.LEFT:
            new_x,new_y = old_x - 1, old_y  # left
        elif act == ChipAction.RIGHT:
            new_x,new_y  = old_x + 1, old_y   # right
        elif act == ChipAction.UP:
            new_x,new_y  = old_x, old_y - 1 # up
        else:
            #act == ChipAction.DOWN:
            new_x,new_y  = old_x, old_y + 1  # down
        #print(f"old_x:{old_x}, old_y:{old_y}, new_x:{new_x}, new_y:{new_y}")
        #if new_post out of matrix
        if (
                new_x < 0
                or new_x >= self._cols
                or new_y < 0
                or new_y >= self._rows
                or self._state[new_x,new_y] != 0):

            print("Invalid move")
            return False
        else:
            #free the old position
            self._state[old_x, old_y] = 0
            #occupy the new position
            self._positions[player] = (new_x, new_y)
            self._state[new_x, new_y] = player

            return True

    def route_to_magic_state(self, player: int):
        '''
        :param player:
        :return: the length to magic state
        '''
        px,py = self._positions[player]
        # use dfs to find the shortest path to magic state(value that equal to -1)

        path_len,path = bfs_find_target(self._state, px, py)
        return path_len

    def all_path_len(self):
        '''
        :return: all path to magic state
        '''
        path_len = []
        for i in range(1,self._num_qubits+1):
            px, py = self._positions[i+1]
            len, path  = bfs_find_target(self._state, px, py)
            path_len.append(len)
        return path_len

    def distance_to_others(self,player):
        '''
        :param player:
        :return: the distance to other qubits
        '''
        px, py = self._positions[player]
        distance = []
        for i in range(1,self._num_qubits+1):
            if i == player:
                continue
            x, y = self._positions[i]
            distance.append(abs(px - x) + abs(py - y))
        #sum distance
        return sum(distance)

    def __str__(self):
        return self._state.__str__()

    @property
    # start from index 1
    def position(self):
        return self._positions

    @property
    def state(self):
        return  self._state
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
    chip = Chip(5,5)
    print(chip)
    print(chip.state)
    print(chip.position)
    for i in range(3):
        player = random.randint(1, chip._num_qubits)
        act  = random.randint(0, 3)
        chip.move(player, act)
    print(chip.state)
    print(chip.position)

    print('routing length  = ', chip.route_to_magic_state(1))
    print('length  = ', chip.distance_to_others(4))

    # rewards = {}
    # for i in range(1, chip._num_qubits + 1):
    #     distance = chip.distance_to_others(i)
    #     print(distance)
    #     r = math.log(distance +1, 2)-3.5
    #     rewards.update({f'agent_{i}': r})
    #
    # print(rewards)
