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
   return  np.array([(8.0, 9.0), (4.0, 8.0), (3.0, 5.0), (5.0, 7.0), (2.0, 8.0), (2.0, 7.0), (7.0, 8.0),
                     (4.0, 9.0), (6.0, 9.0), (8.0, 10.0), (5.0, 7.0), (1.0, 4.0), (3.0, 6.0), (8.0, 8.0),
                     (8.0, 10.0), (3.0, 8.0), (5.0, 7.0), (6.0, 6.0), (2.0, 10.0), (5.0, 9.0), (6.0, 9.0),
                     (6.0, 10.0), (8.0, 9.0), (4.0, 7.0), (4.0, 5.0), (5.0, 7.0), (6.0, 10.0), (8.0, 8.0),
                     (7.0, 9.0), (4.0, 7.0), (6.0, 8.0), (7.0, 7.0), (8.0, 9.0), (2.0, 7.0), (2.0, 8.0),
                     (2.0, 6.0), (7.0, 8.0), (5.0, 7.0), (6.0, 7.0), (4.0, 6.0), (2.0, 4.0), (6.0, 9.0),
                     (9.0, 9.0), (4.0, 7.0), (5.0, 8.0), (3.0, 6.0), (4.0, 4.0), (2.0, 8.0), (6.0, 8.0),
                     (10.0, 10.0), (9.0, 10.0), (8.0, 9.0), (3.0, 4.0), (3.0, 9.0), (6.0, 8.0), (6.0, 7.0),
                     (3.0, 5.0), (5.0, 9.0), (3.0, 9.0), (9.0, 9.0), (2.0, 10.0), (3.0, 7.0), (4.0, 7.0),
                     (7.0, 8.0), (1.0, 4.0), (4.0, 8.0), (4.0, 9.0), (7.0, 9.0), (5.0, 9.0), (6.0, 7.0),
                     (8.0, 10.0), (5.0, 8.0), (3.0, 9.0), (4.0, 9.0), (8.0, 10.0), (6.0, 10.0), (3.0, 6.0),
                     (2.0, 10.0), (4.0, 7.0), (7.0, 10.0), (2.0, 8.0), (2.0, 10.0), (6.0, 8.0), (5.0, 9.0),
                     (5.0, 9.0), (6.0, 8.0), (1.0, 4.0), (7.0, 7.0), (7.0, 7.0), (5.0, 7.0), (2.0, 10.0),
                     (4.0, 7.0), (4.0, 9.0), (6.0, 7.0), (2.0, 6.0), (6.0, 7.0), (6.0, 10.0), (7.0, 8.0),
                     (7.0, 7.0), (8.0, 10.0)]).astype(int)


def get_heat_map():
   return np.array([[0., 0., 0., 0., 0.,0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0.,0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0.,0., 0., 0., 0., 0.],
     [0.5, 0.16666667, 0.16666667, 0.16666667, 0.,0., 0., 0., 0., 0.],
     [0., 0., 0.33333333, 0.16666667, 0., 0., 0., 0., 0., 0.],
     [0., 0.33333333, 0.5, 0.16666667, 0., 0.16666667, 0., 0., 0., 0.],
     [0., 0.33333333, 0.16666667, 1., 1., 0.83333333, 0.66666667, 0., 0., 0.],
     [0., 0.66666667, 0.16666667, 0.33333333, 0.33333333, 0.83333333, 0.66666667, 0.33333333, 0., 0.],
     [0., 0., 0.5, 0.66666667, 0.83333333, 0.5, 0.33333333, 0.66666667, 0.33333333, 0.],
     [0., 0.83333333, 0., 0., 0., 0.66666667, 0.16666667, 0.83333333, 0.16666667, 0.16666667]]).astype(float)


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

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
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


def resize_2d_matrix(matrix, target_size):
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
    img_resized = img.resize(target_size, Image.LANCZOS)

    # 3. 转回 NumPy 矩阵
    resized_matrix = np.array(img_resized)

    # 如果是浮点型输入，转换回 [0, 1] 范围
    if matrix.dtype == np.float64 or matrix.dtype == np.float32:
        resized_matrix = resized_matrix.astype(np.float32) / 255.0

    return resized_matrix
if __name__ == "__main__":
    # gates = get_gates(10)
    gates  = get_gates_fixed()
    heatmap = convert_gates_to_heat_map(10, 10, gates)
    plot_heatmap_data(heatmap)
    #pprint(heatmap)
    # plot_heatmap_data(heatmap)
    # plot_heatmap_data(resize_2d_matrix(heatmap,(7,7)))

    cleaned = [tuple(float(x) for x in pair) for pair in gates]
    print(cleaned)
    # resize = resize_2d_matrix(heatmap,(7,7))
    print(repr(heatmap))