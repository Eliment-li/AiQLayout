import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

from utils.file.file_util import get_root_dir

plt.rcParams["font.family"] = "Arial"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 13

# 示例数据
data = np.array([
    [1.0, 0.962573, 0.926548, 0.89187, 0.85849, 0.82636, 0.795432, 0.765662, 0.737006],
    [1.0, 0.965656, 0.932491, 0.900465, 0.869539, 0.839675, 0.810837, 0.782989, 0.756098],
    [1.0, 0.953678, 0.909502, 0.867373, 0.827194, 0.788877, 0.752335, 0.717486, 0.684251],
    [1.0, 0.963722, 0.928761, 0.895067, 0.862597, 0.831304, 0.801146, 0.772082, 0.744073],
    [1.0, 0.977581, 0.955665, 0.934241, 0.913296, 0.892822, 0.872806, 0.853239, 0.83411],
])
data *=100

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 柱状参数
_x = np.arange(data.shape[0])
_y = np.arange(data.shape[1])
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()
z = np.zeros_like(x)
dx = dy = 0.5
dz = data.T.ravel()

# 颜色映射
norm = Normalize(vmin=data.min(), vmax=data.max())
cmap = plt.get_cmap('viridis_r')
colors = cmap(norm(dz))

bars = ax.bar3d(x, y, z, dx, dy, dz, color=colors, edgecolor='k', alpha=0.95, linewidth=0.7)

cbaxes = fig.add_axes([0.19, 0.25, 0.02, 0.45])  # [left, bottom, width, height]
# 颜色条放在右侧
mappable = ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array(data)
cbar = fig.colorbar(mappable, ax=ax,shrink=0.6, aspect=15, cax=cbaxes)  # Associate the colorbar with the 3D axes
cbar.outline.set_visible(False)
# cbar.set_label('Value', labelpad=8)

# 轴标签
ax.set_xlabel('Layout', labelpad=25)
ax.set_ylabel(r'Error rate ($\times 2.0\times 10^{-8}$)', labelpad=10)
ax.set_zlabel(r'Success rate ($\%$)', labelpad=10)

# 添加xticklabels和yticklabels
ax.set_xticks(_x + dx / 2)
ax.set_xticklabels(['Grid     ', 'Compact_1', 'Compact_2', 'Liner    ', 'QAgent'], rotation=10, ha='right')
ax.set_yticks(_y + dy / 3 -1)
ax.set_yticklabels([0,1,2,3,4,5,6,7,8], rotation=45, ha='right')
ax.invert_xaxis()
ax.invert_yaxis()

# 背景添加网格
# ax.grid(True)
plt.savefig(get_root_dir()+'/results/fig/Success.png',dpi = 500)
plt.show()