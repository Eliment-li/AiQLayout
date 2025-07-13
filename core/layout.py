from copy import deepcopy
from enum import Enum
from pathlib import Path

import numpy as np

from utils.file.csv_util import rootdir
from utils.file.excel_util import ExcelUtil
from utils.file.file_util import get_root_dir

rootdir = Path(get_root_dir())

class QubitState(Enum):
    ANCILLA = 0  # free to use
    MAGIC = -1 # magic state
    BROKEN = -2 #broken, unavailable

class ChipLayoutType(Enum):
    EMPTY = 'EMPTY' #do nothing,do not put any extra qubit on the chip,
    GRID = 'GRID'
    GIVEN = 'GIVEN'
    RANDOM = 'RANDOM'
    COMPACT_1 = 'COMPACT_1'
    COMPACT_2 = 'COMPACT_2'
    LINER_1 = "LINER_1"

class ChipLayout():

    def __init__(self,rows:int,cols:int,layout_type:ChipLayoutType,num_qubits:int):
        self.rows = rows
        self.cols = cols
        self.num_qubits=num_qubits
        self.max_num_qubits=None

        self.broken_channel = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.state = np.zeros((self.rows, self.cols), dtype=int)

        self.layout_type = layout_type
        if layout_type is not None and layout_type == ChipLayoutType.GRID:
            self.dynamic_set_layout()

        #self._init_magic_state()
        # self.init_broken_patch()

    def clean_invalud_qubits(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.state[i][j] > self.num_qubits:
                    self.state[i][j] = 0

    def set_state(self, layout):
        self.state = deepcopy(layout)
        #reset the magic state after set state,because the state should only contain qubits,not magic state or broken qubits



    def dynamic_set_layout(self):
        assert self.layout_type == ChipLayoutType.GRID, "Dynamic layout is designed only be set for GRID layout type."
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


def read_layout_from_xlsx(layout_type:ChipLayoutType):
    file_path = rootdir/"assets"/"chip_layout.xlsx"
    # switch(layout_type):
    #     case ChipLayoutType.EMPTY:
    #         pass
    state = None
    match layout_type:
        case ChipLayoutType.COMPACT_1:
            state = ExcelUtil.read_sheet_to_array(file_path,"COMPACT_1")
        case ChipLayoutType.COMPACT_2:
            state = ExcelUtil.read_sheet_to_array(file_path,"COMPACT_2")
        case ChipLayoutType.LINER_1:
            state = ExcelUtil.read_sheet_to_array(file_path,"LINER_1")
    return state



def get_layout(layout_type,rows,cols, num_qubits,given_state=None):
    if layout_type == ChipLayoutType.GIVEN:
        state = given_state
    elif layout_type == ChipLayoutType.GRID:
        state = np.zeros((rows, cols), dtype=int)
        qubit = 1
        i = 1
        while i < rows - 1:
            j = 1
            while j < cols - 1:
                if qubit > num_qubits:
                    break
                if state[i][j] == 0:
                    state[i][j] = qubit
                    qubit += 1
                j += 2
            i += 2
    elif layout_type == ChipLayoutType.EMPTY:
        state = np.zeros((rows, cols), dtype=int)
    else:
        state = read_layout_from_xlsx(layout_type)


    layout = ChipLayout(
        rows=rows,
        cols=cols,
        layout_type=layout_type,
        num_qubits=num_qubits,
    )
    if layout_type==ChipLayoutType.COMPACT_1:
        assert rows==12 and cols==12, "Compact layout is designed for 12x12 chip."
    elif layout_type==ChipLayoutType.COMPACT_2:
        assert rows == 18 and cols == 18, "Compact layout is designed for 18x18 chip."
    elif layout_type==ChipLayoutType.LINER_1:
        assert rows == 17 and cols == 17, "LINER_1 layout is designed for 17x17 chip."

    layout.set_state(state)
    layout._init_magic_state()
    layout.clean_invalud_qubits()
    return layout


if __name__ == '__main__':
    layout = get_layout(layout_type = ChipLayoutType.GRID, rows=25, cols=25, num_qubits=100)
    print(layout)