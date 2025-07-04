from pprint import pprint

import numpy as np
from PIL._imaging import display
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.visualization import array_to_latex

def demo():
    # 定义标准 T† 门的矩阵
    t_dagger = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

    # z = np.array([[1, 0], [0, -1]], dtype=complex)  # Z 门矩阵
    # t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # T 门矩阵
    #
    # print(t_dagger)
    # print(z*t*z)

    # 构造电路T dagger = Z-S-T
    qc = QuantumCircuit(1)
    qc.z(0)  # T 门
    qc.s(0)
    qc.t(0)

    # # 提取电路的酉矩阵
    circuit_unitary = Operator(qc).data
    print(circuit_unitary)
    print(t_dagger)

    # 验证是否等价（忽略全局相位）
    if np.allclose(circuit_unitary, t_dagger):
        print("\n电路等价于 Tdagger！")
    else:
        # 检查是否仅差全局相位
        global_phase = np.angle(circuit_unitary[0, 0] / t_dagger[0, 0])
        scaled_unitary = circuit_unitary * np.exp(-1j * global_phase)
        if np.allclose(scaled_unitary, t_dagger):
            print(f"\n电路等价于 T†（允许全局相位 {global_phase / np.pi:.2f}π）！")
        else:
            print("\n电路不等价于 T†。")


def reshape_to_2d(arr, n):
    """
    将一维数组转换为二维数组，每行n个元素

    参数:
        arr (list): 输入的一维数组
        n (int): 每行的元素数量

    返回:
        list: 二维数组
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n必须是正整数")

    return [arr[i:i + n] for i in range(0, len(arr), n)]
if __name__ == '__main__':
    import redis
    r = redis.Redis(host='127.0.0.1', port=6379)
    r.flushall()