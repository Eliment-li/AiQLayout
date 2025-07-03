import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

import matplotlib.ticker as ticker

from utils.file.csv_util import to_dataframe
from utils.file.file_util import get_root_dir

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# from matplotlib import font_manager
#
# # 列出可用字体
# available_fonts = sorted([f.name for f in font_manager.fontManager.ttflist])
# print(available_fonts)
mpl.rcParams['font.size'] = 24
label_size = 24
line_width = 1.0
v6=[]
v7=[]
dpi = 500
# Custom formatter function
def time_formatter(x,pos):
    if x <3600:
        # Format as hours and minutes if more than 60 minutes
        return f'{(x/60).__round__(1)}min'
    else:
        # Format as minutes otherwise
        return f'{(x/3600).__round__(1)}hr'
# 定义格式化函数，将秒转换为小时格式
def seconds_to_hours(x, pos):
    # 将秒转换为小时
    hours = x / 3600
    return f'{hours:.1f}'


fig, axs = plt.subplots(2, 1, figsize=(10, 8))

def plot1():
    ax = axs[0]
    data_path = f'assets\\data\\BRB_policy_loss.csv'
    group1,group2 = get_data(data_path)

    for i  in range(len(group1)):
        data = group1[i]
        label='disable BRB' if i == 0 else None
        ax.plot(data,color='#1565c0',linewidth=line_width, label=label)
    for i  in range(len(group2)):
        data = group2[i]
        label = 'enable BRB' if i == 0 else None
        ax.plot(data,color='#df6172',linewidth=line_width, label=label)

    # 设置y轴为对数刻度
    #ax.set_yscale('log',base=10)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 0.02):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    #ax.set_xlabel('Training Iteration',fontsize = label_size)
    ax.set_ylabel('Policy Loss',fontsize = label_size)

    # ustom formatter

    #ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    # 设置 x 轴的主刻度为每 1800 秒（0.5 小时）
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))


    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

    ax.legend(loc='upper right', fontsize='small')

#total loss
def plot2():
    ax = axs[1]
    data_path = f'assets\\data\\BRB_vf_loss.csv'
    group1,group2 = get_data(data_path)
    x = range(0, len(group1[0]))
    for i  in range(len(group1)):
        data = group1[i]
        label='disable BRB' if i == 0 else None
        ax.plot(x,data,color='#1565c0',linewidth=line_width, label=label)
    for i  in range(len(group2)):
        data = group2[i]
        label = 'enable BRB' if i == 0 else None
        ax.plot(x,data,color='#df6172',linewidth=line_width, label=label)

    # 设置y轴为对数刻度
    #ax.set_yscale('log',base=10)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 0.2):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    ax.set_xlabel('Training Iteration',fontsize = label_size)
    ax.set_ylabel('Value Function Loss',fontsize = label_size)

    # ustom formatter

    #ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # 使用 FuncFormatter 应用自定义格式
    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(seconds_to_hours))

    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

    ax.legend(loc='upper right', fontsize='small')

# sample_steps
##The throughput of sampled environmental steps per second,
def plot3():
    ax = axs[0]
    xv6,yv6,xv7,yv7 = get_data('sample_time_s',x_index = 'Step')
    # 创建折线图
    ax.plot(xv6, yv6,color='#1565c0',linewidth=line_width, label='PPO')
    ax.plot(xv7, yv7,color='#df6172',linewidth=line_width, label='RR-PPO')
    ax.set_ylim(bottom=85)
    ax.set_ylim(top=190)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 10):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    ax.set_xlabel('Steps',fontsize = label_size)
    ax.set_ylabel('Sample Time (s)',fontsize = label_size)

    # ustom formatter
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.legend(loc='upper right', fontsize='medium')

#Time in seconds this iteration took to run
def plot4():
    ax = axs[1]
    xv6, yv6, xv7, yv7 = get_data('train_iter_s', x_index='Step')
    # 创建折线图
    ax.plot(xv6, yv6,color='#1565c0',linewidth=line_width, label='PPO')
    ax.plot(xv7, yv7,color='#df6172',linewidth=line_width, label='RR-PPO')
    ax.set_ylim(bottom=85)
    ax.set_ylim(top=190)
    y_min, y_max = ax.get_ylim()
    # 循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 10):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', linewidth=line_width, zorder=0)

    # plt.title('amplitude_estimation')
    ax.set_xlabel('Steps',fontsize = label_size)
    ax.set_ylabel('Iteration Time (s) ',fontsize = label_size)

    # ustom formatter
    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.legend(loc='upper right', fontsize='medium')

def plot_t1():
    plot1()
    plot2()
    plt.tight_layout()
    plt.savefig(get_root_dir()+'/data/fig/t1.png',dpi = dpi)
    plt.show()

def plot_t2():
    plot3()
    plot4()
    plt.tight_layout()
    plt.savefig(get_root_dir()+'/data/fig/t2.png',dpi = dpi)
    plt.show()

def get_data(path):
    data=to_dataframe(relative_path=path)
    group1=[]
    group2=[]
    #iter by column and get index for each column
    i = 0
    for column in data.columns:
        #skip the step column
        if column == 'step':
            continue
        col_data=data[column].values
        col_data =replace_nan_with_average(col_data)
        if i % 2 == 0:
            group1.append(col_data)
        else:
            group2.append(col_data)
        i+=1
    return group1,group2


def replace_nan_with_average(arr):
    #Replace NaN values in the array with the average of the previous and next values.

    arr = np.array(arr, dtype=float)  # Ensure the input is a numpy array of floats

    # Iterate through the array and replace NaN values
    for i in range(len(arr)):
        if np.isnan(arr[i]) or arr[i]=='nan':
            prev_value = arr[i - 1] if i > 0 else None
            next_value = arr[i + 1] if i < len(arr) - 1 else None

            # Calculate the average of previous and next values
            if prev_value is not None and next_value is not None and not np.isnan(prev_value) and not np.isnan(
                    next_value):
                arr[i] = (prev_value + next_value) / 2
            elif prev_value is not None and not np.isnan(prev_value):  # Only previous value exists
                arr[i] = prev_value
            elif next_value is not None and not np.isnan(next_value):  # Only next value exists
                arr[i] = next_value
            else:  # Both are NaN or missing, leave the value as NaN
                arr[i] = np.nan

    return arr



if __name__ == '__main__':
    # plot_t2()
    group1,group2 = get_data(f'assets\\data\\BRB_policy_loss.csv')
    # print(group1)
    # print(group2)
    plot1()
    plot2()
    plt.show()
    # print(replace_nan_with_average(group1[0]))