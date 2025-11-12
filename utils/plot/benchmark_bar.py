import os
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator  # 添加 MultipleLocator
from matplotlib.ticker import FuncFormatter  # 新增导入

from utils.file.excel_util import ExcelUtil
from utils.file.file_util import get_root_dir

mpl.rcParams['font.family'] = ['Arial']
mpl.rcParams['font.size'] = 34
rootdir = get_root_dir()
sep = os.path.sep
#9个柱状图

def get_data():
    # # get data
    path = r'D:\paper\dac2026\experiment/sum_up.xlsx'
    sheets, dfs = ExcelUtil.read_by_sheet(path)
    data={}
    labels_2d = [

    ]
    # pharse data

    def parse_series_value(v):
        """
        输入可能为：
         - 单个数值（int/float）或字符串数字 -> 返回 mean, std (std=0)
         - 逗号分隔的数字字符串 '1,2,3' 或 '1, 2,3' -> 返回均值和标准差
         - 空值或无法解析 -> 返回 np.nan, 0
        """
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan, 0.0
        # 如果已经是数值
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v), 0.0
        # 转为字符串并尝试按逗号分割
        s = str(v).strip()
        if s == '':
            return np.nan, 0.0
        parts = [p.strip() for p in s.split(',') if p.strip() != '']
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except:
                # 无法解析则忽略该部分
                continue
        if len(vals) == 0:
            return np.nan, 0.0
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    for sheet in sheets:
        df = dfs[sheet]
        #get 0:4 row
        df = df.iloc[0:4, :]

        qubits_number = df['qubits'].tolist()
        qubits_number = np.array(qubits_number).astype(int)
        # 从字符串中提取出该线路的比特数量 qnn/qnn_indep_qiskit_5.qasm-> 5
        #print(labels)
        labels_2d.append(qubits_number)
        rl = df['rl_distance'].tolist()
        grid = df['grid_distance'].tolist()

        # 解析每一项，得到 mean 列表和 std 列表
        rl_means = []
        rl_stds = []
        for v in rl:
            m, s = parse_series_value(v)
            rl_means.append(m)
            rl_stds.append(s)

        grid_means = []
        grid_stds = []
        for v in grid:
            m, s = parse_series_value(v)
            grid_means.append(m)
            grid_stds.append(s)

        # 保证 numpy 数组类型为 float，便于绘图
        rl_means = np.array(rl_means, dtype=float)
        rl_stds = np.array(rl_stds, dtype=float)
        grid_means = np.array(grid_means, dtype=float)
        grid_stds = np.array(grid_stds, dtype=float)

        data.update({sheet:(rl_means, rl_stds, grid_means, grid_stds)})

    print('===data===')
    print(data)
    return data,labels_2d

def plot():
    data,labels_2d = get_data()

    title = [
        'Quantum Fourier Transformation',
        'Quantum Walk',
        'Deutsch-Jozsa Algorithm',
        'Quantum Neural Network',
    ]

    labels=['a','b','c','d']
    # 设置每个柱子的宽度
    bar_width = 0.04

    # 设置柱子的位置
    index = np.arange(len(labels))*0.12

    # 创建3x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # figsize可以根据需要调整

    plt.subplots_adjust(hspace=0.4,wspace = 0.4)
    # 遍历数据和子图网格，绘制柱状图
    cnt = 0
    all = 0
    for i, (group_name, (group1_means, group1_stds, group2_means, group2_stds)) in enumerate(data.items()):
        # 计算子图的行和列索引
        row = i // 2
        col = i % 2

        # 获取当前子图的axes对象
        ax = axes[row, col]
        # 设置子图边框宽度
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)  # 将边框宽度设置为2
            spine.set_edgecolor('grey')  # 设置边框颜色

        # 绘制第一组数据的柱状图，使用 mean 作为高度，用 std 作为 yerr
        ax.bar(index, group1_means,  bar_width, color = '#5370c4', label=f'{group_name} 1', hatch='', edgecolor='black', zorder = 2,
               yerr=group1_stds, error_kw={'elinewidth':2, 'capsize':6, 'capthick':2, 'ecolor':'black'})

        # 绘制第二组数据的柱状图，使用 mean 作为高度，用 std 作为 yerr
        ax.bar(index + bar_width, group2_means, bar_width, color = '#f16569', label=f'{group_name} 2', hatch='/', edgecolor='black', zorder = 2,
               yerr=group2_stds, error_kw={'elinewidth':2, 'capsize':6, 'capthick':2, 'ecolor':'black'})

        fontsize = 28
        # 添加图例
        if i==0:
            ax.legend(['QAgent ','Ecmas+'], loc='upper left')

        # 设置横轴的标签
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(index + bar_width / 1)
        ax.set_xticklabels(labels_2d[i])

        # 设置科学计数法格式
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # 将科学计数法标志放在左侧靠上
        ax.yaxis.get_offset_text().set_position((-0.05, 0.9))  # 调整位置
        ax.yaxis.get_offset_text().set_fontsize(fontsize)  # 设置字体大小

        # 最左侧两幅图添加纵坐标说明（文字与纵轴平行）
        if col == 0:
            ax.set_ylabel('Ancilla Qubit Cost', fontsize=fontsize, rotation=90, labelpad=18, va='center')

        # 最底部一行的两幅图添加横坐标说明
        if row == 1:
            ax.set_xlabel('Qubits', fontsize=fontsize+6)

        if i in range(6,9):
            ax.set_xlabel('Qubits' ,fontsize = fontsize+6)
        # 设置图表的标题
        ax.set_title(title[i], fontsize = fontsize+6)
        # 设置纵轴刻度间隔为0.5
        #ax.yaxis.set_major_locator(MultipleLocator(0.5*group2_means.max()/10))
        # 显示背景网格
        ax.grid(True, which='both', axis='y', linestyle='-', linewidth=2,zorder = 0)

        # 调整子图之间的间距
        plt.tight_layout()

        path = rootdir + sep + 'results' + sep + 'fig' + sep + 'benchmarkBar.png'
    plt.savefig(path,dpi=300)
    # 显示图表
    plt.show()

if __name__ == '__main__':
    plot()