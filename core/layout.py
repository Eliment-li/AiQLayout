from enum import Enum

import numpy as np


class QubitState(Enum):
    ANCILLA = 0  # free to use
    MAGIC = -1 # magic state
    BROKEN = -2 #broken, unavailable

class QubitLayoutType(Enum):
    EMPTY = 0 #do nothing,do not put any extra qubit on the chip,
    GRID = 1
    GIVEN = 2
    RANDOM = 3
    COMPACT_1 = 4

class ChipLayout():

    def __init__(self,rows:int,cols:int,layout_type:QubitLayoutType,max_num_qubits:int,state:list,num_qubits:int):
        self.rows = rows
        self.cols = cols
        self.num_qubits
        self.max_num_qubits=max_num_qubits
        self.layout_type=layout_type
        self.state
        if layout_type == QubitLayoutType.GRID:
            self.dynamic_set_layout()
        self.available_pos=self.get_available_qubits()
    def get_available_qubits(self):
        '''
        :return: a list of available qubits
        '''
        available_qubits = [()] * self.max_num_qubits
        for i in range(self.rows):
            for j in range(self.cols):
                index = self.layout[i][j]
                if index > 0:
                    available_qubits[index] = (i,j)
        return available_qubits

    def dynamic_set_layout(self):
        assert self.layout_type == QubitLayoutType.GRID, "Dynamic layout is designed only be set for GRID layout type."
        self.state = np.zeros((self.rows,self.cols), dtype=int)
        qubit = 1
        i = 1
        while i < self.rows - 1:
            j = 1
            while j < self.cols - 1:
                if qubit > self.num_qubits:
                    return
                if self.state[i][j] == 0 and self._broken_channel[i][j] == 0:
                    self.state[i][j] = qubit
                    self._qubits_channel[i][j] = qubit
                    self._position_mask[qubit - 1][i][j] = 1
                    self.q_pos.append((i, j))
                    qubit += 1
                j += 2
            i += 2
        if qubit < self.num_qubits:
            print(f"Warning: only {qubit - 1} qubits are placed, but {self.num_qubits} are required.")
            # TODO try random laytout for rest qubit
        else:
            return



COMPACT_Layout_1 = ChipLayout(
    rows=10,
    cols=10,
    layout_type=QubitLayoutType.COMPACT_1,
    max_num_qubits=36,
    layout=[[0, -1, 0, 0, -1, 0, 0, -1, 0, 0],
            [0, 1, 2, 0, 5, 6, 0, 9, 10, -1],
            [-1, 3, 4, 0, 7, 8, 0, 11, 12, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 13, 14, 0, 17, 18, 0, 21, 22, -1],
            [-1, 15, 15, 0, 19, 20, 0, 23, 24, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 25, 26, 0, 29, 30, 0, 33, 34, -1],
            [-1, 27, 28, 0, 31, 32, 0, 35, 36, 0],
            [0, 0, -1, 0, 0, -1, 0, 0, -1, 0]]
)
