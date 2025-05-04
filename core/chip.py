#representing a quantum computing device chip
import math
import random

import numpy as np
from config import ConfigSingleton

from core.routing import bfs_find_target

args = ConfigSingleton().get_args()

class Chip():
    def __init__(self, name=''):
        self._name = name
        self._n_qubits = args.num_qubits
        self._positions=[]
        self._state=np.zeros((args.chip_size_h,args.chip_size_w), dtype=np.int32)
        # magic state
        self._magic_state = []
        self._init_magic_state()
        self._init_qubits_layout()
        self.last_distance=[]

    def random_init(self):
        # vaild value start from _position[1] , -1 only for occupy
        self._positions = [-1]
        i = 1
        while i <= self._n_qubits:
            x = random.randint(0, args.chip_size_h - 1)
            y = random.randint(0, args.chip_size_w - 1)
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
        for r in range(args.chip_size_h):
            for c in range(args.chip_size_w):
                if self._state[r][c] == 0:
                    self._positions.append((r, c))
                    self._state[r][c] = len(self._positions)-1
                    if len(self._positions)-1 == self._n_qubits:
                        return

    def reset(self):
       self._state = np.zeros((args.chip_size_h, args.chip_size_w), dtype=np.int32)
       self._init_qubits_layout()

    def _init_magic_state(self):
        self._magic_state = [
            (0, 0),
            (0, args.chip_size_w - 1),
            (args.chip_size_h - 1, 0),
            (args.chip_size_h - 1, args.chip_size_w - 1),
        ]

        for x, y in self._magic_state:
            self._state[x][y] = -1

    def move(self, player: int, act:int):
        old_x,old_y = self._positions[player]
        if act == 0:
            new_x,new_y = old_x - 1, old_y  # left
        elif act == 1:
            new_x,new_y  = old_x + 1, old_y   # right
        elif act == 2:
            new_x,new_y  = old_x, old_y - 1 # up
        elif act == 3:
            new_x,new_y  = old_x, old_y + 1  # down
        #print(f"old_x:{old_x}, old_y:{old_y}, new_x:{new_x}, new_y:{new_y}")
        #if new_post out of matrix
        if (
                new_x < 0
                or new_x >= args.chip_size_w
                or new_y < 0
                or new_y >= args.chip_size_h
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
        for i in range(1,self._n_qubits+1):
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
        for i in range(1,self._n_qubits+1):
            if i == player:
                continue
            x, y = self._positions[i]
            distance.append(abs(px - x) + abs(py - y))
        #sum distance
        return sum(distance)

    def __str__(self):
        return f"Chip(name={self._name}"

    @property
    def position(self):
        return self._positions

    @property
    def state(self):
        return  self._state

    def plot(self):
        pass

#test code
if __name__ == '__main__':
    chip = Chip("chip")
    print(chip)
    print(chip.state)
    print(chip.position)
    for i in range(1000):
        player = random.randint(1, chip._n_qubits)
        act  = random.randint(0, 3)
        chip.move(player, act)
    print(chip.state)
    print(chip.position)

    print('routing length  = ', chip.route_to_magic_state(1))
    print('length  = ', chip.distance_to_others(4))

    # rewards = {}
    # for i in range(1, chip._n_qubits + 1):
    #     distance = chip.distance_to_others(i)
    #     print(distance)
    #     r = math.log(distance +1, 2)-3.5
    #     rewards.update({f'agent_{i}': r})
    #
    # print(rewards)
