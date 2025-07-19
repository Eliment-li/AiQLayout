import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)

# 主图
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label='y=sin(x)')

# 添加局部放大图
# 参数[左, 下, 宽, 高]，单位是axes fraction
axins = inset_axes(ax, width="40%", height="30%", loc='upper right', borderpad=2)

# 在放大图上画相同的数据
axins.plot(x, y)

# 设置放大区域的x轴和y轴范围
x1, x2 = 2, 4  # x轴放大范围
y1, y2 = y[(x > x1) & (x < x2)].min(), y[(x > x1) & (x < x2)].max()  # y轴自动适应
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# 去掉放大图的刻度标签
axins.set_xticklabels([])
axins.set_yticklabels([])

# 在主图上画出放大区域的矩形，并连线到inset
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

ax.legend()
plt.show()
