from copy import deepcopy
from enum import Enum

import numpy as np

from config import ConfigSingleton


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
    COMPACT_2 = 5

class ChipLayout():

    def __init__(self,rows:int,cols:int,layout_type:QubitLayoutType,num_qubits:int):
        self.rows = rows
        self.cols = cols
        self.num_qubits=num_qubits
        self.max_num_qubits=None

        self.broken_channel = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.state = np.zeros((self.rows, self.cols), dtype=int)

        self.layout_type = layout_type
        if layout_type is not None and layout_type == QubitLayoutType.GRID:
            self.dynamic_set_layout()

        self._init_magic_state()
        # self.init_broken_patch()

    def clean_invalud_qubits(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.state[i][j] > self.num_qubits:
                    self.state[i][j] = 0

    def set_state(self, layout):
        self.state = deepcopy(layout)
        #reset the magic state after set state,because the state should only contain qubits,not magic state or broken qubits


    # def set_valid_qubits(self,layout):
    #     '''
    #     :return: a list of available qubits
    #     '''
    #     available_qubits = [()] * self.num_qubits
    #     for i in range(self.rows):
    #         for j in range(self.cols):
    #             qubit = layout[i][j]
    #             if qubit > 0:
    #                 available_qubits[qubit-1] = (i,j)
    #     return available_qubits

    def dynamic_set_layout(self):
        assert self.layout_type == QubitLayoutType.GRID, "Dynamic layout is designed only be set for GRID layout type."
        qubit = 1
        i = 1
        while i < self.rows - 1:
            j = 1
            while j < self.cols - 1:
                if qubit > self.num_qubits:
                    return
                if self.state[i][j] == 0 and self.broken_channel[i][j] == 0:
                    self.state[i][j] = qubit
                    # self._qubits_channel[i][j] = qubit
                    # self._position_mask[qubit - 1][i][j] = 1
                    # self.q_pos.append((i, j))
                    qubit += 1
                j += 2
            i += 2
        if qubit < self.num_qubits:
            print(f"Warning: only {qubit - 1} qubits are placed, but {self.num_qubits} are required.")
            # TODO try random laytout for rest qubit
        else:
            return

    def init_broken_patch(self):
        broken = [(5,5),(5,6),(5,7),(5,8),(6,5),(7,5)]
        for x, y in broken:
            #self.state[x][y] = QubitState.BROKEN.value
            self.state[x][y] = QubitState.BROKEN.value
            self.broken_channel[x][y] = QubitState.BROKEN.value
            #TODO action mask

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



def get_layout(name,rows,cols, num_qubits):
    layout = ChipLayout(
        rows=rows,
        cols=cols,
        layout_type=name,
        num_qubits=num_qubits,
    )
    if name==QubitLayoutType.COMPACT_1:
        assert rows==12 and cols==12, "Compact layout is designed for 12x12 chip."
        layout.set_state( np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 2, 0, 5, 6, 0, 9, 10, 0, 0],
                           [0, 0, 3, 4, 0, 7, 8, 0, 11, 12, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 13, 14, 0, 17, 18, 0, 21, 22, 0, 0],
                           [0, 0, 15, 16, 0, 19, 20, 0, 23, 24, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 25, 26, 0, 29, 30, 0, 33, 34, 0, 0],
                           [0, 0, 27, 28, 0, 31, 32, 0, 35, 36, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    elif name==QubitLayoutType.COMPACT_2:
        assert rows == 18 and cols == 18, "Compact layout is designed for 18x18 chip."
        layout.set_state(np.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0],
       [  0,   0,   1,   2,   0,   5,   6,   0,   9,  10,   0,  13,  14,
          0,  17,  18,   0,   0],
       [  0,   0,   3,   4,   0,   7,   8,   0,  11,  12,   0,  15,  16,
          0,  19,  20,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0],
       [  0,   0,  21,  22,   0,  25,  26,   0,  29,  30,   0,  33,  34,
          0,  37,  38,   0,   0],
       [  0,   0,  23,  24,   0,  27,  28,   0,  31,  32,   0,  35,  36,
          0,  39,  40,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0],
       [  0,   0,  41,  42,   0,  45,  46,   0,  49,  50,   0,  53,  54,
          0,  57,  58,   0,   0],
       [  0,   0,  43,  44,   0,  47,  48,   0,  51,  52,   0,  55,  56,
          0,  59,  60,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0],
       [  0,   0,  61,  62,   0,  65,  66,   0,  69,  70,   0,  73,  74,
          0,  77,  78,   0,   0],
       [  0,   0,  63,  64,   0,  67,  68,   0,  71,  72,   0,  75,  76,
          0,  79,  80,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0],
       [  0,   0,  81,  82,   0,  85,  86,   0,  89,  90,   0,  93,  94,
          0,  97,  98,   0,   0],
       [  0,   0,  83,  84,   0,  87,  88,   0,  91,  92,   0,  95,  96,
          0,  99, 100,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0]]))



    layout._init_magic_state()
    layout.clean_invalud_qubits()
    return layout

    # def _random_init_qubits_layout(self):
    #     # vaild value start from _position[1] , -1 only for occupy
    #     i = 1
    #     while i <= self.num_qubits:
    #         x = random.randint(0, self._rows - 1)
    #         y = random.randint(0, self._cols - 1)
    #         if self.state[x][y] == 0 and self._broken_channel[x][y]== 0:
    #
    #             self.state[x][y] = i
    #             self._qubits_channel[x][y] = i
    #             self._position_mask[i-1][x][y] = 1
    #
    #             self.q_pos.append((x, y))
    #             i += 1
    #         else:
    #             continue


if __name__ == '__main__':
    layout = get_layout(name = QubitLayoutType.COMPACT_2, rows=18, cols=18, num_qubits=100)
    print(layout)