import os
from datetime import datetime

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Switching to Agg backend.')
    matplotlib.use('Agg')  # 无头模式
else:
    matplotlib.use('TkAgg')  # 有图形界面时使用 TkAgg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator  # 用于设置整数刻度
from pathlib import Path
from utils.file_util import get_root_dir



def plot_reward(data, save=True):
    print('plot reward ...')
    '''
    # Example:
    data = [
        [10, 20, 30, 40, 50],
        [0, 342, 200, 500, 600],
        [3, 4, 5, 6, 7]
    ]
    plot_reward(data)
    '''

    # 创建一个图形和一个坐标轴
    fig, ax1 = plt.subplots()

    # Generate a colormap
    num_lines = len(data)
    colors = plt.cm.viridis(np.linspace(0, 1, num_lines))

    x1 = np.arange(len(data[0]))
    y1 = data[0]



    ax1.plot(x1, y1, label='reward', color='#5370c4', marker='o')
    ax1.set_xlabel('step')
    ax1.set_ylabel('reward', color='#5370c4')
    for x, y in zip(x1, y1):
        # Annotate each point with its value
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        # Draw a vertical line from each point to the x-axis
        ax1.axvline(x=x, ymin=0, ymax=(y - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0]), color=colors[0],
                    linestyle='--', linewidth=0.5)

    x2 = np.arange(len(data[1]))
    y2 = data[1]
    #创建第二个坐标轴，共享x轴
    ax2 = ax1.twinx()
    ax2.plot(x2, y2, label='distance', color='#f16569', marker='v')
    ax2.set_ylabel('distance', color='#f16569')
    for x, y in zip(x2, y2):
        # Annotate each point with its value
        ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # set xaxis to integer
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Add labels and legend
    plt.title('reward')
    #plt.show()
    if save:
        print('save figure ...')
        time_id = datetime.now().strftime('%Y-%m-%d_%H-%M')
        path = Path(get_root_dir()) / 'results' / (time_id + '.png')
        plt.savefig(path)


if __name__ == '__main__':
    data = [
        [10, 20, 30, 40, 50],
        [200, 200, 200, 500, 600],
    ]

    plot_reward(data)