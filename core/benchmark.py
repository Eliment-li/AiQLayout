import ast
import math
from pathlib import Path
from pprint import pprint

import numpy as np

from core.chip import Chip, compute_depth, benchmark_layouts
from core.layout import ChipLayoutType, get_layout, QubitState, ChipLayout
from core.routing import a_star_path
from utils.file.excel_util import ExcelUtil
from utils.file.file_util import get_root_dir
from utils.ls_instructions import get_heat_map
from utils.route_util import bfs_route

rootdir = get_root_dir()
def compute_success_rate(num_qubits, heat_map, chip,error_rate):
    base_rate = 1 - error_rate
    final_rate = 1
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
            # sum_dist += (cnt * dist)
            op_rate = math.pow(base_rate, dist)
            op_rate = math.pow(op_rate, cnt)
            final_rate = final_rate * op_rate

    return final_rate

def success_rate(layout:ChipLayout,num_qubits: int = 0, size: int=0,heat_map=None):
    layout_type = ChipLayoutType(layout.layout_type)
    layout = get_layout(layout_type=layout_type, rows=size, cols=size,num_qubits=num_qubits)
    # layout = ChipLayout(rows=args.chip_rows,cols=args.chip_cols,layout_type = ChipLayoutType.GRID,num_qubits=self.num_qubits)#get_layout(name = ChipLayoutType.GRID, rows=args.chip_rows, cols=args.chip_cols, num_qubits=self.num_qubits)
    temp_chip = Chip(rows=size, cols=size, num_qubits=num_qubits,layout=layout)
    #dist  = compute_dist_v2(num_qubits, heat_map, temp_chip)
    success_rate  = compute_success_rate(num_qubits, heat_map, temp_chip)
    return success_rate

def str_to_array(array_str):
    # 去掉前缀 'array(' 和最后的 ')'
    array_str_clean = array_str.replace('array(', '').rstrip(')')

    # 用 ast.literal_eval 转成 list
    array_list = ast.literal_eval(array_str_clean)

    # 转成 numpy 数组
    array_np = np.array(array_list)
    return array_np

def benchmark_qagent_success_rate():
    # # get data
    path = r'D:\AAAI2026\experiment/qft/sumUp.xlsx'
    sheets, dfs = ExcelUtil.read_by_sheet(path)

    layout = dfs['layout']
    layout = layout.iloc[0:10, :]
    nqubits = layout['qubits'].tolist()
    nqubits = np.array(nqubits).astype(int)
    states = layout['state'].tolist()
    result =[]
    print(nqubits)
    for j in range(10):
        error_rate = j*2e-8
        row_data = []
        #for i in range(len(nqubits-1)):
        for i in [20]:
            #n = nqubits[i]
            n = i
            s = str_to_array(states[8])
            size = len(s[0])
            file = Path(f'assets/circuits/qft/LSI_qftentangled_indep_qiskit_{n}.lsi')
            heat_map = get_heat_map(file_path=rootdir / file)
            # layout = get_layout(layout_type=ChipLayoutType.GRID, rows=size, cols=size, num_qubits=n,
            #                     given_state=None)
            size = 17
            layout = get_layout(layout_type=ChipLayoutType.LINER_1, rows=size, cols=size, num_qubits=n,
                                given_state=None)
            temp_chip = Chip(rows=size, cols=size, num_qubits=n, layout=layout)
            sr = compute_success_rate(num_qubits=n, heat_map=heat_map, chip=temp_chip,error_rate=error_rate)
            sr = round(sr, 6)
            row_data.append(sr)
            result.append(sr)

        print(row_data)
    print(result)




def temp():
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

    # for j in range(len(chip_size)):
    #     layout = LAYOUT[j]
    #     size = chip_size[j]
    #     print(layout)
    #     for i in range(4):
    #         num_qubits = lsi_size[i]
    #         file = Path(f'assets/circuits/qft/LSI_qftentangled_indep_qiskit_{num_qubits}.lsi')
    #         heat_map = get_heat_map(file_path=rootdir / file)
    #         rate = success_rate(layout_type=ChipLayoutType(layout), num_qubits=num_qubits, size=size,heat_map=heat_map)
    #         print(rate)

    state = np.array([[ 0,  0,  0, -1,  0,  0, -1,  0,  0, -1,  0,  0],
       [ 0,  0,  0, 19,  0,  0,  3,  0,  0,  0,  0, -1],
       [-1,  0,  0,  0,  0,  0, 10,  0, 18,  0,  0,  0],
       [ 0,  0, 21,  0,  5, 20,  0,  0,  0,  0,  0,  0],
       [ 0, 13, 11,  0,  0,  0,  0,  0,  0, 15,  0, -1],
       [-1,  0,  0, 16, 14,  1,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0, 22,  0, 12,  0,  9,  0, 23,  0],
       [ 0,  0,  0,  0,  4,  0,  0,  0,  8,  0,  0, -1],
       [-1,  0,  0,  6,  0,  0,  0, 25,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  7,  0,  0,  0, 17,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0, -1],
       [-1,  0,  0, -1,  0, 24, -1,  0,  0, -1,  0,  0]])
    size = 12
    num_qubits = 25
    file = Path(f'assets/circuits/qft/LSI_qftentangled_indep_qiskit_{num_qubits}.lsi')
    heat_map = get_heat_map(file_path=rootdir / file)
    layout = get_layout(layout_type=ChipLayoutType.GIVEN, rows=size, cols=size, num_qubits=num_qubits,given_state=state)
    temp_chip = Chip(rows=size, cols=size, num_qubits=num_qubits, layout=layout)
    sr = compute_success_rate(num_qubits=num_qubits, heat_map=heat_map,chip = temp_chip)
    print(sr)
#test code
if __name__ == '__main__':
    benchmark_qagent_success_rate()

    # grid_success =[
    #
    # ]
    #
    # rl_success = []
    # error_rate1 = 1- np.array(grid_success)
    # error_rate1 *=100
    # print(repr(error_rate1))
