from pprint import pprint

import numpy as np
from PIL._imaging import display
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.visualization import array_to_latex

# 定义标准 T† 门的矩阵
t_dagger = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

# z = np.array([[1, 0], [0, -1]], dtype=complex)  # Z 门矩阵
# t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # T 门矩阵
#
# print(t_dagger)
# print(z*t*z)

# 构造电路T dagger = Z-S-T
qc = QuantumCircuit(1)
qc.z(0)      # T 门
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
        print(f"\n电路等价于 T†（允许全局相位 {global_phase/np.pi:.2f}π）！")
    else:
        print("\n电路不等价于 T†。")