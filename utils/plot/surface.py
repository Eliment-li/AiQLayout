import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RegularGridInterpolator

plt.rcParams["font.family"] = "Arial"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 13

def plot_3d_surface_smooth(data,
                           cbar_label='Fidelity Improvement(%)',
                           save_path=None):
    # 原始网格
    y = np.arange(data.shape[0])
    x = np.arange(data.shape[1])
    # 插值函数
    interp_func = RegularGridInterpolator((y, x), data, method='linear')
    # 新的更密集网格
    ynew = np.linspace(0, data.shape[0] - 1, 9)
    xnew = np.linspace(0, data.shape[1] - 1, 9)
    x_grid, y_grid = np.meshgrid(xnew, ynew)
    points = np.stack([y_grid.ravel(), x_grid.ravel()], axis=-1)
    data_smooth = interp_func(points).reshape(9, 9)
    # 归一化用于着色
    norm = Normalize(vmin=0, vmax=25)
    n_data_smooth = norm(data_smooth)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        x_grid, y_grid, data,
        rstride=1, cstride=1,
        facecolors=plt.cm.viridis(data),
        shade=False
    )

    cbaxes = fig.add_axes([0.17, 0.25, 0.02, 0.45])
    mappable = ScalarMappable(cmap='viridis')
    mappable.set_array(data.flatten())
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=15, cax=cbaxes)
    cbar.outline.set_visible(False)
    cbar.set_label(cbar_label)

    ax.set_ylabel('T1 Relaxation Time (µs)', labelpad=10)
    ax.set_xlabel('T2 Relaxation Time \n($18\\% \\leq T2/T1 \\leq 120\\%$)', labelpad=10)
    ax.set_zlabel('Value', labelpad=10)

    x_ticks = [0, 5, 10, 16]
    x_labels = ['18', '48', '78', '120']
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels)

    y_ticks = [0, 5, 10, 15, 20]
    y_labels = ['10', '20', '30', '40', '50']
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels(y_labels)

    ax.view_init(elev=17, azim=-60)

    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()

# 用法示例
if __name__ == '__main__':
    # np.random.seed(0)
    # data = np.random.uniform(0, 25, (21, 17))
    data = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.80236811e-06,
         8.52452908e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         3.80236811e-06, 3.80236811e-06, 3.80236811e-06, 3.80236811e-06,
         1.68577990e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.80236811e-06,
         3.80236811e-06, 3.80236811e-06, 3.80236811e-06, 7.60473623e-06,
         2.50039925e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.80236811e-06,
         3.80236811e-06, 3.80236811e-06, 7.60473623e-06, 7.60473623e-06,
         3.29680525e-01],
        [0.00000000e+00, 0.00000000e+00, 3.80236811e-06, 3.80236811e-06,
         3.80236811e-06, 7.60473623e-06, 7.60473623e-06, 1.14071043e-05,
         4.07530210e-01],
        [0.00000000e+00, 0.00000000e+00, 3.80236811e-06, 3.80236811e-06,
         7.60473623e-06, 7.60473623e-06, 1.14071043e-05, 1.14071043e-05,
         4.83638410e-01],
        [0.00000000e+00, 3.80236811e-06, 3.80236811e-06, 3.80236811e-06,
         7.60473623e-06, 7.60473623e-06, 1.14071043e-05, 1.52094725e-05,
         5.58039347e-01],
        [0.00000000e+00, 3.80236811e-06, 3.80236811e-06, 3.80236811e-06,
         7.60473623e-06, 1.14071043e-05, 1.52094725e-05, 1.52094725e-05,
         6.30774847e-01]],)
    data = np.log10(data+1e-6)
    plot_3d_surface_smooth(data)
