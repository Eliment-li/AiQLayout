import math

import torch
from torch import nn


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

    return pe



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



# a = torch.zeros((1, 4, 5, 5))
# l = PositionalEncoding2DSpaceOnly(d_model=4)
#
# print(l(a)[0])

pos_encoding = positionalencoding2d(9,9, 8)

print(pos_encoding)