import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable


plt.rcParams["font.family"] = "Arial"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 13

data1 =np.array(
[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999999, 0.977581],
[1.0, 1.0, 1.0, 1.0, 0.999999, 0.999999, 0.999999, 0.999999, 0.955665],
[1.0, 1.0, 1.0, 0.999999, 0.999999, 0.999999, 0.999999, 0.999998, 0.934241],
[1.0, 1.0, 1.0, 0.999999, 0.999999, 0.999999, 0.999998, 0.999998, 0.913296],
[1.0, 1.0, 0.999999, 0.999999, 0.999999, 0.999998, 0.999998, 0.999997, 0.892822],
[1.0, 1.0, 0.999999, 0.999999, 0.999998, 0.999998, 0.999997, 0.999997, 0.872806],
[1.0, 0.999999, 0.999999, 0.999999, 0.999998, 0.999998, 0.999997, 0.999996, 0.853239],
[1.0, 0.999999, 0.999999, 0.999999, 0.999998, 0.999997, 0.999996, 0.999996, 0.83411]]
)



data2=np.array(
[
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.962573],
        [1.0, 1.0, 0.999999, 0.999999, 0.999999, 0.999998, 0.999998, 0.999997, 0.926548],
        [1.0, 0.999999, 0.999999, 0.999998, 0.999998, 0.999997, 0.999997, 0.999996, 0.89187],
        [0.999999, 0.999999, 0.999998, 0.999998, 0.999997, 0.999996, 0.999996, 0.999995, 0.85849],
        [0.999999, 0.999999, 0.999998, 0.999997, 0.999997, 0.999994, 0.999995, 0.999993, 0.82636],
        [0.999999, 0.999999, 0.999998, 0.999997, 0.999996, 0.999993, 0.999994, 0.999992, 0.795432],
        [0.999999, 0.999998, 0.999997, 0.999996, 0.999995, 0.999992, 0.999993, 0.999991, 0.765662],
        [0.999999, 0.999998, 0.999997, 0.999996, 0.999995, 0.999991, 0.999992, 0.999989, 0.737006]
])

# data1 = 1-data1
# data2  = 1-data2
# data1 = np.random.uniform(0, 25, (21, 17))
# data2 = np.random.uniform(0, 25, (21, 17))

all_data = np.concatenate([data1.flatten(), data2.flatten()])
norm = Normalize(vmin=all_data.min(), vmax=all_data.max())
n_data1 = norm(data1)
n_data2 = norm(data2)



# 创建一个新的图形和 3D 轴
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 获取 colormap 实例
cmap = plt.get_cmap('viridis')
# 定义热力图的高度
heights = [10,   30]

# 创建网格坐标
x, y = np.meshgrid(range(data1.shape[1]), range(data1.shape[0]))

print(repr(n_data1))
# 创建第一个热力图
surf1 = ax.plot_surface(x, y, np.full_like(data1, heights[0]), rstride=1, cstride=1, facecolors=cmap(n_data1), shade=False)
# 创建第二个热力图
surf2 = ax.plot_surface(x, y, np.full_like(data2, heights[1]), rstride=1, cstride=1, facecolors=cmap(n_data2), shade=False)

# 创建第三个热力图
#surf3 = ax.plot_surface(x, y, np.full_like(data3, heights[2]), rstride=1, cstride=1, facecolors=plt.cm.viridis(data3), shade=False)
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# cbaxes = inset_axes(ax, width="3%", height="50%", loc='center left')

cbaxes = fig.add_axes([0.17, 0.25, 0.02, 0.45])  # [left, bottom, width, height]
# 添加颜色条
mappable = ScalarMappable(cmap='viridis',norm = norm)
mappable.set_array(all_data)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=15,cax=cbaxes)  # 调整 shrink 和 aspect 参数
cbar.outline.set_visible(False)
cbar.set_label('Fidelity Improvement',)
# 设置轴标签
ax.set_ylabel('t1',labelpad=10)
ax.set_xlabel('t2',labelpad=10)
# ax.set_zlabel('Bits')

# 设置Z轴的刻度位置和标签
z_ticks = [10, 30,40]
z_labels = ['6 Bits', '8 Bits', '']
ax.set_zticks(z_ticks)
ax.set_zticklabels(z_labels)
ax.zaxis.set_tick_params(pad=8)

# 设置x轴刻度和标签
x_ticks = [0,5,10,16]
x_labels = ['18', '48','78','120']
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(x_labels)

# 设置Y轴刻度和标签
y_ticks = [0,5,10,15,20]
y_labels = ['10', '20','30','40','50']
# ax.set_yticks(y_ticks)
# ax.set_yticklabels(y_labels)


# 设置标题
# ax.set_title('3D Stacked Heatmaps')
# 设置视角
ax.view_init(elev=17, azim=-60)
# 调整轴的范围以更好地展示热力图
# ax.set_xlim(0, data1.shape[1]-1)
# ax.set_ylim(0, data1.shape[0]-1)
# ax.set_zlim(min(heights)-5, max(heights)+5)
# plt.savefig(FileUtil.get_root_dir()+'/data/fig/fidelity.png',dpi = 500)

# 显示图形
plt.show()