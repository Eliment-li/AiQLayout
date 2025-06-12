import os
from collections import defaultdict

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sympy import pprint


def plot_heatmap_data(heatmap_data):
    x_vals = [i+1 for i in range(len(heatmap_data))]
    y_vals = [i+1  for i in range(len(heatmap_data[0]))]
    plt.figure(figsize=(10, 10))

    # 使用绿色色系
    im = plt.imshow(heatmap_data, cmap='Greens', interpolation='nearest',
                   extent=[min(x_vals)-0.5, max(x_vals)+0.5,
                          min(y_vals)-0.5, max(y_vals)+0.5],
                   origin='lower')

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('cnt', rotation=270, labelpad=15)

    # 设置坐标轴为整数
    plt.xticks(np.arange(min(x_vals), max(x_vals)+1, 1))
    plt.yticks(np.arange(min(y_vals), max(y_vals)+1, 1))

    # 添加网格线
    plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # 标签和标题
    plt.xlabel('X (x ≤ y)')
    plt.ylabel('Y (x ≤ y)')
    plt.title(' (stander x ≤ y)')
    # 显示图形

    plt.show()

###
def get_gates_fixed():
   # return np.array([(3.0, 4.0), (3.0, 9.0), (7.0, 9.0), (9.0, 10.0), (5.0, 8.0), (5.0, 9.0), (7.0, 9.0), (3.0, 7.0), (2.0, 2.0),
   #   (4.0, 6.0), (9.0, 10.0), (7.0, 9.0), (1.0, 5.0), (6.0, 9.0), (4.0, 6.0), (4.0, 6.0), (3.0, 5.0), (2.0, 8.0),
   #   (2.0, 7.0), (6.0, 6.0), (2.0, 8.0), (2.0, 7.0), (4.0, 5.0), (6.0, 8.0), (3.0, 9.0), (5.0, 6.0), (5.0, 8.0),
   #   (3.0, 6.0), (8.0, 9.0), (5.0, 6.0), (7.0, 9.0), (8.0, 9.0), (8.0, 9.0), (2.0, 9.0), (3.0, 4.0), (5.0, 6.0),
   #   (7.0, 9.0), (5.0, 9.0), (9.0, 10.0), (4.0, 7.0), (9.0, 9.0), (2.0, 5.0), (8.0, 10.0), (3.0, 5.0), (3.0, 6.0),
   #   (7.0, 7.0), (2.0, 6.0), (8.0, 9.0), (5.0, 8.0), (6.0, 8.0), (6.0, 9.0), (6.0, 6.0), (3.0, 7.0), (4.0, 4.0),
   #   (6.0, 10.0), (7.0, 9.0), (7.0, 10.0), (5.0, 6.0), (3.0, 6.0), (5.0, 7.0), (6.0, 8.0), (3.0, 7.0), (4.0, 9.0),
   #   (6.0, 7.0), (10.0, 10.0), (8.0, 10.0), (8.0, 8.0), (2.0, 5.0), (8.0, 9.0), (3.0, 10.0), (4.0, 9.0), (7.0, 9.0),
   #   (3.0, 5.0), (5.0, 10.0), (6.0, 7.0), (4.0, 9.0), (3.0, 7.0), (7.0, 8.0), (2.0, 5.0), (8.0, 9.0), (1.0, 9.0),
   #   (5.0, 6.0), (3.0, 10.0), (2.0, 9.0), (5.0, 10.0), (4.0, 5.0), (3.0, 3.0), (5.0, 8.0), (6.0, 7.0), (2.0, 9.0),
   #   (6.0, 8.0), (7.0, 9.0), (3.0, 8.0), (5.0, 10.0), (4.0, 7.0), (5.0, 6.0), (1.0, 8.0), (7.0, 8.0), (1.0, 9.0),
   #   (7.0, 9.0)]).astype(int)
   return  np.array([(9.0, 13.0), (5.0, 14.0), (5.0, 11.0), (6.0, 8.0), (10.0, 11.0), (8.0, 13.0), (4.0, 6.0), (10.0, 15.0),
                     (13.0, 14.0), (5.0, 9.0), (10.0, 12.0), (8.0, 9.0), (6.0, 7.0), (12.0, 17.0), (9.0, 9.0), (5.0, 16.0),
                     (6.0, 12.0), (6.0, 6.0), (9.0, 17.0), (10.0, 11.0), (7.0, 9.0), (4.0, 8.0), (13.0, 15.0), (7.0, 8.0),
                     (4.0, 12.0), (10.0, 14.0), (11.0, 16.0), (14.0, 16.0), (4.0, 9.0), (3.0, 4.0), (8.0, 13.0), (4.0, 14.0),
                     (6.0, 13.0), (5.0, 14.0), (8.0, 13.0), (6.0, 9.0), (2.0, 5.0), (13.0, 18.0), (4.0, 8.0), (5.0, 17.0),
                     (2.0, 14.0), (11.0, 17.0), (10.0, 10.0), (8.0, 12.0), (9.0, 15.0), (10.0, 18.0), (12.0, 14.0), (4.0, 9.0),
                     (9.0, 9.0), (13.0, 18.0), (16.0, 17.0), (10.0, 14.0), (8.0, 18.0), (4.0, 17.0), (5.0, 10.0), (2.0, 7.0),
                     (14.0, 14.0), (4.0, 9.0), (6.0, 7.0), (6.0, 10.0), (14.0, 15.0), (7.0, 9.0), (3.0, 7.0), (11.0, 18.0),
                     (3.0, 11.0), (8.0, 15.0), (10.0, 11.0), (8.0, 17.0), (8.0, 14.0), (3.0, 7.0), (16.0, 16.0), (8.0, 14.0),
                     (9.0, 9.0), (7.0, 10.0), (16.0, 16.0), (7.0, 10.0), (4.0, 17.0), (8.0, 17.0), (4.0, 4.0), (8.0, 12.0),
                     (13.0, 17.0), (8.0, 13.0), (6.0, 10.0), (10.0, 16.0), (7.0, 8.0), (2.0, 16.0), (10.0, 11.0), (6.0, 10.0),
                     (4.0, 14.0), (8.0, 13.0), (6.0, 7.0), (3.0, 14.0), (2.0, 11.0), (12.0, 13.0), (13.0, 13.0), (10.0, 17.0),
                     (6.0, 8.0), (11.0, 11.0), (7.0, 9.0), (5.0, 12.0)]).astype(int)


def  get_gates(num_qubits:int,size = 200,format=None ):
    # 设置正态分布的均值和标准差
    mean = 10  # 均值
    std_dev = 5  # 标准差
    from scipy.stats import truncnorm

    a = (1 - mean) / std_dev  # 下限标准化
    b = (num_qubits - mean) / std_dev  # 上限标准化
    trunc_data = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size)
    trunc_data = np.round(trunc_data)
    # 生成正态分布的随机数
    qubits =trunc_data
    i = 0
    gates = []
    while i < len(qubits):
        if qubits[i] < qubits[i + 1]:
            gates.append((qubits[i], qubits[i + 1]))
        else:
            gates.append((qubits[i +1], qubits[i]))
        i += 2
    if format == 'heatmap':
        return  convert_gates_to_heat_map()
    return gates

def convert_gates_to_heat_map(x,y,gates):
    # 统计标准化坐标的频率
    coord_counts = defaultdict(int)
    for coord in gates:
        coord_counts[tuple(coord)] += 1

    # 获取所有 x 和 y 值（x ≤ y）
    x_vals =[i for i in range(1,x+1)] #sorted({x for x, y in coord_counts.keys()})
    y_vals =[i for i in range(1,y+1)]  #sorted({y for x, y in coord_counts.keys()})

    # 创建热力图数据矩阵（只包含 x ≤ y 的部分）
    heatmap_data = np.zeros((len(y_vals), len(x_vals)))

    # 填充数据
    for (x, y), count in coord_counts.items():
        x_idx = x_vals.index(x)
        y_idx = y_vals.index(y)
        heatmap_data[y_idx, x_idx] = count

    #normalize the heatmap_data
    heatmap_data = heatmap_data / np.max(heatmap_data) if np.max(heatmap_data) != 0 else heatmap_data

    return heatmap_data


if __name__ == "__main__":
    gates = get_gates(18)
    # gates  = get_gates_fixed()
    heatmap = convert_gates_to_heat_map(18, 18, gates)
    plot_heatmap_data(heatmap)
    cleaned = [tuple(float(x) for x in pair) for pair in gates]
    print(cleaned)
