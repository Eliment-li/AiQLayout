import math

class ZScoreNormalizer:
    def __init__(self):
        self.mean = 0  # 均值
        self.var = 0   # 方差
        self.n = 0     # 数据点数量

    def update(self, x):
        self.n += 1
        old_mean = self.mean
        # 更新均值
        self.mean += (x - self.mean) / self.n
        # 更新方差
        self.var += (x - old_mean) * (x - self.mean)
        # 计算标准差
        std = math.sqrt(self.var / self.n) if self.n > 1 else 0
        # 返回 Z-Score 标准化值
        return (x - self.mean) / std if std > 0 else 0

# 示例数据流
data = [10, 20, 15, 30, 25]

# 初始化动态归一化器
normalizer = ZScoreNormalizer()

# 动态处理数据
print("Original -> Z-Score Normalized:")
for x in data:
    z_score = normalizer.update(x)
    print(f"{x} -> {z_score}")
