import re

import numpy as np
import random
from matplotlib import pyplot as plt


def generate_irregular_obstacle(size, complexity=0.75, density=0.75):
    """
    生成不规则形状的障碍物

    参数:
    size -- 障碍物的大致尺寸 (width, height)
    complexity -- 形状复杂度 (0-1)
    density -- 填充密度 (0-1)

    返回:
    二维数组表示的不规则形状障碍物
    """
    width, height = size
    shape = np.zeros((height, width), dtype=int)

    # 确保障碍物至少有一个单元格
    shape[random.randint(0, height - 1)][random.randint(0, width - 1)] = 1

    # 随机生成障碍物形状
    for _ in range(int(complexity * min(width, height))):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        shape[y][x] = 1

        # 随机扩展
        for _ in range(int(density * min(width, height))):
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                shape[ny][nx] = 1
                x, y = nx, ny

    return shape


def generate_map(n, obstacle_sizes, num_obstacles=1, max_attempts=100):
    """
    生成带有随机不规则障碍物的n×n地图

    参数:
    n -- 地图大小(n×n)
    obstacle_sizes -- 障碍物尺寸列表 [(w1,h1), (w2,h2), ...]
    num_obstacles -- 要放置的障碍物数量
    max_attempts -- 放置障碍物的最大尝试次数

    返回:
    numpy数组表示的地图，0表示空地，1表示障碍物
    """
    map_grid = np.zeros((n, n), dtype=int)
    placed = 0
    attempts = 0

    while placed < num_obstacles and attempts < max_attempts:
        # 随机选择一个障碍物尺寸
        width, height = random.choice(obstacle_sizes)

        # 确保障碍物不会超出地图边界
        if width > n or height > n:
            attempts += 1
            continue

        # 随机选择障碍物的左上角位置
        x = random.randint(0, n - width)
        y = random.randint(0, n - height)

        # 生成不规则形状障碍物
        obstacle = generate_irregular_obstacle((width, height))

        # 检查该区域是否已被占用
        overlap = np.sum(map_grid[y:y + height, x:x + width] * obstacle)
        if overlap == 0:
            # 放置障碍物
            map_grid[y:y + height, x:x + width] += obstacle
            placed += 1

        attempts += 1

    return map_grid


def visualize_map(map_grid):
    """可视化地图"""
    plt.figure(figsize=(8, 8))
    plt.imshow(map_grid, cmap='binary', interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.show()



'''test code'''
# 示例使用
if __name__ == "__main__":
    # n = 20  # 地图大小
    # obstacle_sizes = [(3, 3), (2, 4), (5, 2), (4, 4)]  # 可能的障碍物尺寸
    #
    # # 生成带有5个不规则障碍物的地图
    # obstacle_map = generate_map(n, obstacle_sizes, num_obstacles=5)
    #
    # print("生成的地图(0=空地, 1=障碍物):")
    # print(obstacle_map)
    #
    # # 可视化地图
    # visualize_map(obstacle_map)



    # # 示例用法
    # n = 100  # 原始尺寸
    # m = 32  # 目标尺寸
    #
    # # 生成一个随机矩阵（可以是 uint8 或 float32）
    # original_matrix_uint8 = np.random.randint(0, 256, (n, n), dtype=np.uint8)  # 0-255
    # original_matrix_float = np.random.rand(n, n).astype(np.float32)  # 0-1
    #
    # # 缩放
    # resized_uint8 = resize_2d_matrix(original_matrix_uint8, (m, m))
    # resized_float = resize_2d_matrix(original_matrix_float, (m, m))
    #
    # print("Original shape (uint8):", original_matrix_uint8.shape)
    # print("Resized shape (uint8):", resized_uint8.shape)
    # print("Original shape (float):", original_matrix_float.shape)
    # print("Resized shape (float):", resized_float.shape)
    pass

