import os

import matplotlib

qubits = [12,10,7,5,15,6,8,8,10,12,3,13,9,17,9,13,11,
          8,4,10,14,12,5,6,8,9,19,8,17,12,2,10,10,3,10,
          13,15,5,5,2,10,4,5,13,9,9,8,4,1,11,1,13,12,6,10,
          4,17,2,7,6,9,18,14,13,12,11,8,5,9,11,14,9,10,4,15,6,12,15,11,
          5,5,8,10,19,5,9,1,7,6,19,8,12,9,14,10,5,17,8,6,2,9,3,11,5,10,8,
          3,4,4,3,5,13,7,17,15,9,15,9,2,10,14,4,1,16,7,1,4,
          19,5,9,13,3,14,5,16,5,11,10,12,8,5,8,18,4,14,9,10,4,
          14,8,10,14,6,13,5,9,5,1,8,11,11,17,12,11,12,14,4,20,16,
          7,11,7,7,3,5,8,5,4,8,8,2,7,9,9,8,9,7,6,7,9,2,11,15,8,13,
          12,8,1,6,7]


def  get_gates():
    i = 0
    gates = []
    while i < len(qubits):
        gates.append((qubits[i],qubits[i+1]))
        i += 2
    return gates


def generate_gates():
    import numpy as np
    import matplotlib.pyplot as plt

    # 设置正态分布的均值和标准差
    mean = 10  # 均值
    std_dev = 5  # 标准差

    # 生成正态分布的随机数
    random_numbers = np.random.normal(mean, std_dev, 200)

    # 将随机数限制在 0 到 19 的范围，并取整
    bounded_integers = np.clip(random_numbers, 1, 20).astype(int)

    #print bounded_integers,split number with ,
    output_string = ",".join(map(str, bounded_integers))
    print(output_string)

    # 统计每个数字出现的频率
    values, counts = np.unique(bounded_integers, return_counts=True)

    # 绘制柱状图
    plt.bar(values, counts, color='skyblue', edgecolor='black')

    # 添加图标题和轴标签
    plt.title('Distribution of Random Integers (0-19)', fontsize=14)
    plt.xlabel('Integer Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # 设置 x 轴刻度为整数
    plt.xticks(range(1, 21))

    # 显示图表
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()