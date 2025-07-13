import math
from pprint import pprint

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor


def positionalencoding2d( height, width,d_model):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.numpy()



# 示例使用
# x, y = 9, 9
# d_model = 4  # 位置编码的维度
# pos_encoding = positionalencoding2d(9,9, 8)
# # print("位置编码:", pos_encoding)
#
# print(pos_encoding[0])
# print(pos_encoding[1])
# print(pos_encoding[2])
# print(pos_encoding[3])
# print(pos_encoding[4])



class PositionalEncoding(nn.Module):
    """An absolute pos encoding layer."""
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len , embedding_dim]``
        """
        x = x + self.pe[None, :x.size(1)]
        return self.dropout(x)


class PositionalEncodingTransposed(PositionalEncoding):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__(d_model, dropout, max_len)
        self.pe = torch.permute(self.pe, (1, 0))  # [max_len, d_model] to [d_model, max_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, embedding_dim, seq_len]``
        """
        x = x + self.pe[None, :, :x.size(2)]
        return self.dropout(x)


class PositionalEncoding2D(PositionalEncodingTransposed):
    """A 2D absolute pos encoding layer."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__(d_model=d_model // 2, dropout=dropout, max_len=max_len)
        self.d_model_half = d_model // 2
        # self.proj         = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding ="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, gate_color, space , time]``
        """

        p1 = self.pe[None, :, :x.size(2), None]  # space encoding
        p2 = self.pe[None, :, None, :x.size(3)]  # time encoding

        x[:, :self.d_model_half] = x[:, :self.d_model_half] + p1
        x[:, self.d_model_half:] = x[:, self.d_model_half:] + p2

        # x = self.proj(x)

        return self.dropout(x)

class PositionalEncoding2DSpaceOnly(PositionalEncodingTransposed):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__(d_model=d_model, dropout=dropout, max_len=max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, gate_color, space , time]``
        """

        p1 = self.pe[None, :, :x.size(2), None]  # space encoding
        return self.dropout(x + p1)



# a = torch.zeros((1, 1, 5, 5))
# a[0][0][0][0] = 0
# l = PositionalEncoding2DSpaceOnly(d_model=8)
#
# print(l(a)[0])

# pos_encoding = positionalencoding2d(3,3, 4)
#
# print(pos_encoding)
# print(pos_encoding.shape)
#
# a = torch.zeros((1, 3, 3))
# a = a.repeat(4,1,1)
# print(a.shape)
#
# print(a+pos_encoding)

# pprint(positionalencoding2d(15, 15, 4))

def plot_pe(heatmap_data):
    x_vals = [i + 1 for i in range(len(heatmap_data))]
    y_vals = [i + 1 for i in range(len(heatmap_data[0]))]
    plt.figure(figsize=(10, 10))

    # 使用绿色色系
    im = plt.imshow(heatmap_data, cmap='Greys', interpolation='nearest',
                    extent=[min(x_vals) - 0.5, max(x_vals) + 0.5,
                            min(y_vals) - 0.5, max(y_vals) + 0.5],
                    origin='lower')

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('cnt', rotation=270, labelpad=15)

    # 设置坐标轴为整数
    plt.xticks(np.arange(min(x_vals), max(x_vals) + 1, 1))
    plt.yticks(np.arange(min(y_vals), max(y_vals) + 1, 1))

    # 添加网格线
    plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # 标签和标题
    plt.xlabel('X (x ≤ y)')
    plt.ylabel('Y (x ≤ y)')
    plt.title(' (stander x ≤ y)')
    # 显示图形

    plt.show()




def get_2d_sinusoidal_positional_embedding(height: int, width: int, dim: int, base: float = 10000) -> Tensor:
    """return embed (height, width, dim)"""
    assert dim % 4 == 0
    grid_h = torch.arange(height, dtype=torch.float32)
    grid_w = torch.arange(width, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    embed_h = sinusoidal_embedding(grid[0], dim // 2, base=base)
    embed_w = sinusoidal_embedding(grid[1], dim // 2, base=base)
    embed = torch.cat([embed_h, embed_w], dim=-1)
    return embed

def sinusoidal_embedding(idx: Tensor, dim: int, base: float = 10000) -> Tensor:
    """idx (*) -> embed (*, dim)"""
    assert dim % 2 == 0
    half_dim = dim // 2
    freqs = torch.arange(half_dim, dtype=torch.float32) / half_dim
    freqs = torch.exp(-math.log(base) * freqs).to(device=idx.device)
    embed = idx.float()[..., None] * freqs
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=-1)
    return embed
if __name__ == '__main__':
    # pos_encoding = positionalencoding2d(9,9, 8)
    # print("位置编码:", pos_encoding)
    #
    # for i in range(4):
    #     n1 = np.array(pos_encoding[i][0][0])
    #     n2 = np.array(pos_encoding[i][1][0])
    #     print(n1,n2)

    #embed = get_2d_sinusoidal_positional_embedding(7, 7, 4)

    # embed = np.array([
    #     [
    #         [11, 12, 13],
    #         [14, 15, 16],
    #         [17, 18, 19],
    #
    #     ],
    #     [
    #         [21, 22, 23],
    #         [2, 2, 2],
    #         [2, 2, 2],
    #
    #     ],
    #     [
    #         [3, 3, 3],
    #         [3, 3, 3],
    #         [3, 3, 3],
    #
    #     ],
    #     [
    #         [4, 4, 4],
    #         [4, 4, 4],
    #         [4, 4, 4],
    #
    #     ],
    # ])
    # print(embed[...,0])

    embed = positionalencoding2d(7,7, 4)

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(embed[ i])
        plt.title(f'Channel {i}')
    plt.show()
