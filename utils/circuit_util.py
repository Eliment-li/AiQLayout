import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sympy import pprint

from utils.file.file_util import write_to_file, get_root_dir


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



def generate_node_relations(n,count=100, mean=0.2, std_dev=0.3, min_relations=10):
    """
    生成节点关系对

    参数:
    n -- 正节点数量(1到n)
    mean -- 正态分布均值(控制关系密度)
    std_dev -- 正态分布标准差
    min_relations -- 每个节点至少拥有的关系数

    返回:
    包含节点对的列表
    """
    nodes = list(range(1, n + 1)) + [-1]  # 所有节点包括-1
    relations = []

    # # 确保每个节点至少有min_relations个连接
    # for node in nodes:
    #     # 随机选择min_relations个其他节点连接
    #     other_nodes = [n for n in nodes if n != node]
    #     for _ in range(min_relations):
    #         if not other_nodes:
    #             break
    #         partner = random.choice(other_nodes)
    #         relations.append((min(node, partner), max(node, partner)))
    #         other_nodes.remove(partner)
    #
    # # 去重
    # relations = list(set(relations))

    # 基于正态分布添加额外关系
    for k in range(count):
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]
                # 使用正态分布概率决定是否添加关系
                prob = np.random.normal(mean, std_dev)
                prob = max(0, min(1, prob))  # 限制在0-1范围内
                if random.random() < prob:
                    relations.append((min(node1, node2), max(node1, node2)))

    # 再次去重
    #relations = list(set(relations))
    random.shuffle(relations)  # 打乱顺序
    return relations

def gates_to_LSI(gates,num_qubits:int):
    lsi = []
    m_i = 10000
    for i in range(len(gates)):
        (q0,q1) = gates[i]
        if q0 ==-1:
            '''
            RequestMagicState 1000001
            MultiBodyMeasure 0:Z,1000001:X
            MeasureSinglePatch 1000001 Z
            '''
            lsi.append(f'RequestMagicState {m_i}')
            lsi.append(f'MultiBodyMeasure {q1}:Z,{m_i}:X')
            lsi.append(f'MeasureSinglePatch {m_i} Z')
            m_i += 1
        else:
            lsi.append(
                f'MultiBodyMeasure {q0}:Z,{q1}:X'
            )

    content=''
    for instruction in lsi:
        content += instruction + '\n'
        print(instruction)

    return content

    #write to file

def plot_gate_heatmap(nodes, relations,savepath=None):
    size = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    matrix = np.zeros((size, size))
    for a, b in relations:
        i, j = node_index[a], node_index[b]
        matrix[i, j] += 1
        #matrix[j, i] += 1  # 无向
    plt.figure(figsize=(8, 8))
    im =plt.imshow(matrix, cmap='Greens',  interpolation='nearest',origin='lower')

    cbar = plt.colorbar(im,shrink=0.7, aspect=20)
    cbar.set_label('cnt', rotation=270, labelpad=10)
    # 设置坐标轴标签
    labels = [str(node) for node in nodes]
    plt.xticks(range(len(nodes)), labels)
    plt.yticks(range(len(nodes)), labels)

    plt.xlabel("node")
    plt.ylabel("node")
    plt.title("heatmap")
    #plt.colorbar(label="relation")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    #plt.show()

from PIL import Image
import numpy as np
from scipy.ndimage import zoom


def resize_3d_array(array_3d, zoom_factor):
    """
    对一个 3D 数组的每个 2D 数组进行放大或缩小。

    参数:
        array_3d (numpy.ndarray): 输入的 3D 数组，形状为 (depth, height, width)。
        zoom_factor (tuple): 放大倍数，形状为 (zoom_y, zoom_x)，
                             分别表示沿行和列方向的放大倍数。

    返回:
        numpy.ndarray: 放大或缩小后的 3D 数组。
    """
    # 检查输入是否为 3D 数组
    if len(array_3d.shape) != 3:
        raise ValueError("输入的数组必须是 3D 数组，形状为 (depth, height, width)。")

    # 计算放大后的新形状
    depth, height, width = array_3d.shape
    new_height = int(height * zoom_factor[0])
    new_width = int(width * zoom_factor[1])
    new_shape = (depth, new_height, new_width)

    # 初始化放大后的 3D 数组
    array_3d_zoomed = np.zeros(new_shape)

    # 对每个 2D 数组进行放大
    for i in range(depth):
        array_3d_zoomed[i] = zoom(array_3d[i], zoom_factor)

    return array_3d_zoomed


def resize_2d_matrix(matrix, r,c):
    """
    使用 Lanczos 重采样缩放二维矩阵
    :param matrix: 输入矩阵 (n x n)
    :param target_size: 目标尺寸 (m, m)
    :return: 缩放后的矩阵 (m x m)
    """
    # 1. 将矩阵转为 Pillow Image 对象
    if matrix.dtype == np.float64 or matrix.dtype == np.float32:
        # 如果是浮点型，假设范围在 [0, 1]，转为 [0, 255] 的 uint8
        img = Image.fromarray((matrix * 255).astype(np.uint8))
    else:
        # 直接转换（假设是 uint8 或其他兼容类型）
        img = Image.fromarray(matrix)

    # 2. 使用 Lanczos 重采样
    img_resized = img.resize((r,c), Image.LANCZOS)

    # 3. 转回 NumPy 矩阵
    resized_matrix = np.array(img_resized)

    # 如果是浮点型输入，转换回 [0, 1] 范围
    if matrix.dtype == np.float64 or matrix.dtype == np.float32:
        resized_matrix = resized_matrix.astype(np.float32) / 255.0

    return resized_matrix

def generates_random_lsi(qubits_num):
    gates = generate_node_relations(qubits_num, 100)
    content = gates_to_LSI(gates, qubits_num)

    path = Path(get_root_dir()) / 'assets' / 'circuits' / 'random' / f'LSI_random_indep_qiskit_{qubits_num}.lsi'
    write_to_file(path, content)
    nodes = [-1] + list(range(1, qubits_num + 1))  # -1在前

    savepath = Path(get_root_dir()) / 'assets' / 'circuits' /'random'/ f'gates_heatmap_{qubits_num}.png'
    plot_gate_heatmap(nodes, gates,savepath)
    # plot_heatmap_data(resize_2d_matrix(heatmap,(7,7)))
if __name__ == "__main__":
    for i in range(1,6):
         generates_random_lsi(2*i)
