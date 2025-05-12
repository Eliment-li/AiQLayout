import math

class ZScoreNormalizer:
    '''
    #test code


    # 示例数据流
    data = [10, 20, 15, 30, 25]

    # 初始化动态归一化器
    normalizer = ZScoreNormalizer()

    # 动态处理数据
    print("Original -> Z-Score Normalized:")
    for x in data:
        z_score = normalizer.update(x)
        print(f"{x} -> {z_score}")

    '''
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


class CNN:
    @staticmethod
    def calculate_conv_output_size(input_size, conv_spec):
        """
        计算卷积层输出的数据维度

        参数:
            input_size: 输入图像的尺寸 (height, width)
            conv_spec: 卷积层配置列表，每个元素为 [过滤器数量, 卷积核大小, 步幅]

        返回:
            输出数据的维度 (channels, height, width)
       # test code
            conv_spec = [
                [16, 2, 1],  # 过滤器数量，卷积核大小，步幅
                [32, 3, 1],
                [64, 3, 1],
            ]

            input_size = (10, 10)

            final_channels, final_height, final_width = calculate_conv_output_size(input_size, conv_spec)
            print("\n最终输出维度:", (final_channels, final_height, final_width))

        """
        channels = 1  # 初始输入通道数（假设是灰度图像）
        height, width = input_size

        for layer in conv_spec:
            filters, kernel_size, stride = layer
            # 计算输出高度和宽度
            height = (height - kernel_size) // stride + 1
            width = (width - kernel_size) // stride + 1
            channels = filters  # 更新通道数为当前层的过滤器数量

            print(f"层配置 {layer}: 输出尺寸 ({channels}, {height}, {width})")
        print(f'final data dimensions: {channels * height * width}')
        return channels, height, width
