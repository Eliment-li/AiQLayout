import os
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from utils.file.excel_util import ExcelUtil
from utils.file.file_util import get_root_dir

# from utils.file.excel_util import ExcelUtil
# from utils.file.file_util import FileUtil

mpl.rcParams['font.family'] = ['Arial']
mpl.rcParams['font.size'] = 34
rootdir = get_root_dir()
sep = os.path.sep
#9个柱状图

data={}
labels_2d = [

]
title = [
    'Quantum Fourier Transformation',
    'Amplitude Estimation',
    'Quantum Neural Network',
    'Efficient SU2 ansatz',
    'Portfolio Optimization with VQE',
    'Real Amplitudes ansatz',
]
# # get data
path = r'd:/sync/AiQLayout_data/sum_up.xlsx'
sheets, dfs = ExcelUtil.read_by_sheet(path)
# pharse data
for sheet in sheets:
    df = dfs[sheet]
    circuits = df['circuit']
    # 从字符串中提取出该线路的比特数量 qnn/qnn_indep_qiskit_5.qasm-> 5
    qubits_number = list(map(lambda x: ''.join(re.findall(r'\d', x)), circuits))
    #print(labels)
    labels_2d.append(qubits_number)
    rl = df['rl']
    grid = df['grid']
    data.update({sheet:(rl,grid)})


data = {
    'g1':([1,2,3],[4,5,6]),
    'g2':([1,2,3],[4,5,6]),
    'g3':([1,2,3],[4,5,6]),
    'g4':([1,2,3],[4,5,6])

}
labels=['a','b','c']
labels_2d=[labels,labels,labels,labels,labels,labels]

# 设置每个柱子的宽度
bar_width = 0.04

# 设置柱子的位置
index = np.arange(len(labels))*0.12

# 创建3x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(14, 14))  # figsize可以根据需要调整

plt.subplots_adjust(hspace=0.4,wspace = 0.4)
# 遍历数据和子图网格，绘制柱状图
cnt = 0
all = 0
for i, (group_name, (group1, group2)) in enumerate(data.items()):
    # 计算子图的行和列索引
    row = i // 2
    col = i % 2

    g1= np.array(group1)#.values
    g2 = np.array(group2)#.values
    res = 1-(g1/g2)
    for v in res:
        all+=1
        if v> 0.2:
            cnt+=1
        print(v)
    #print(1-(g1/g2))

    # 获取当前子图的axes对象
    ax = axes[row, col]
    # 设置子图边框宽度
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 将边框宽度设置为2
        spine.set_edgecolor('grey')  # 设置边框颜色

    # 绘制第一组数据的柱状图
    ax.bar(index, group1,  bar_width,color = '#5370c4',label=f'{group_name} 1',hatch='--', edgecolor='black',zorder = 0)

    # 绘制第二组数据的柱状图
    ax.bar(index + bar_width, group2, bar_width,color = '#f16569', label=f'{group_name} 2',hatch='/', edgecolor='black',zorder = 1)
    #ax.bar(index + bar_width*2, group3, bar_width,color = '#95c978', label=f'{group_name} 3',hatch='//', edgecolor='black')
    fontsize = 26
    # 添加图例
    if i==0:
        ax.legend(['QAgent ','Grid'], loc='upper left')

    # 设置横轴的标签
    ax.set_xticks(index + bar_width / 1)
    ax.set_xticklabels(labels_2d[i])
    # if i % 2 ==0:
    #     ax.set_ylabel('Depth', fontsize = fontsize)
    if i in range(6,9):
        ax.set_xlabel('Qubits' ,fontsize = fontsize)
    # 设置图表的标题
    ax.set_title(title[i], fontsize = fontsize+6)
    # 显示背景网格
    ax.grid(True, which='both', axis='y', linestyle='-', linewidth=1,zorder = 0)
# print(cnt/all)
    # 调整子图之间的间距
    plt.tight_layout()

    path = rootdir + sep + 'results' + sep + 'fig' + sep + 'benchmarkBar.png'
plt.savefig(path,dpi=300)
# 显示图表
plt.show()

