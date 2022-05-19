from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import logging
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model_panop/mha.py --> class PositionalEncoding -->  def __init__()')
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MHAAcrossFeat(nn.Module):

    def __init__(self, embed_dim=128,num_heads=8, mode=0, is_pe=True, sum_dim=2): # cfg.MHA.WITH_PE
        super(MHAAcrossFeat, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model_panop/mha.py --> class MHAAcrossFeat -->  def __init__()')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.with_pe = is_pe
        self.mode = mode
        self.mha = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.sum_dim = sum_dim
        if self.with_pe:
            self.pe = PositionalEncoding(self.embed_dim)

    def forward(self, x, y):
        '''
        ax - torch.Size([984, 1, 128])
        aw_y - torch.Size([1, 984, 984])
        wa_ax.shape - torch.Size([1, 984, 128])
        wa_ax.shape - torch.Size([984, 1, 128])
        wa_ax.shape - torch.Size([24, 41, 1, 128])
        x.shape - torch.Size([24, 41, 1, 128])
        ay.shape - torch.Size([984, 1, 128])
        ay.shape - torch.Size([1, 984, 128])
        wa_ay.shape - torch.Size([1, 984, 128])
        wa_ay.shape - torch.Size([984, 1, 128])
        wa_ay.shape -torch.Size([24, 41, 1, 128])
        y - torch.Size([24, 41, 1, 128])
        '''
        N, C, H, W = x.shape    # 1, 128, H, W
        x = x.permute(2, 3, 0, 1)  # (H, W, 1, 128)
        x = x.view(-1, N, C)  #
        y = y.permute(2, 3, 0, 1)  # (H, W, 1, 128)
        y = y.view(-1, N, C)  # (H*W, 1, 128)
        if self.with_pe:
            x = self.pe(x)
            y = self.pe(y)
        if self.mode == 0:
            out, _ = self.mha(y, x, x)  # (H*W, 1, 128)
        elif self.mode == 1:
            out, _ = self.mha(x, y, y)  # (H*W, 1, 128)
        elif self.mode == 2:
            ax, aw_x = self.mha(y, x, x)
            ay, aw_y = self.mha(x, y, y)
            out = torch.cat((ax, ay), dim=2)

        elif self.mode == 3 or self.mode == 4:
            ax, aw_x = self.mha(x, x, x)
            ay, aw_y = self.mha(y, y, y)

            aw_x = torch.transpose(aw_x, 1, 2)
            aw_y = torch.transpose(aw_y, 1, 2)

            ax = ax.permute(1, 0, 2)
            wa_ax = torch.bmm(aw_y, ax)
            wa_ax = wa_ax.permute(1, 0, 2)
            wa_ax = wa_ax.view(H, W, N, C)
            x = x.view(H, W, N, C)
            ay = ay.permute(1, 0, 2)
            wa_ay = torch.bmm(aw_x, ay)
            wa_ay = wa_ay.permute(1, 0, 2)
            wa_ay = wa_ay.view(H, W, N, C)
            y = y.view(H, W, N, C)
            if self.mode == 3:
                x = x + wa_ax
                y = y + wa_ay
                out = torch.cat((x, y), dim=3)
            elif self.mode == 4:
                ax = ax.permute(1, 0, 2)
                ax = ax.view(H, W, N, C)
                ay = ay.permute(1, 0, 2)
                ay = ay.view(H, W, N, C)
                ax = ax + wa_ax
                ay = ay + wa_ay
                out = torch.cat((ax, ay), dim=3)

        if self.mode == 0 or self.mode == 1:
            out = out.view(H, W, N, C)
        # elif self.mode == 2 or self.mode == 3 or self.mode == 4:
        #     out = out.view(H, W, N, C*2)
        out = out.permute(2, 3, 0, 1)
        return out

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model_panop/mha.py --> class MHAAcrossFeat -->  get_10x_lr_params()')
        b = []
        b.append(self.mha.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


class MHAWithinFeat(nn.Module):
    def __init__(self, cfg, inpd, outd):
        super(MHAWithinFeat, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model_panop/mha.py --> class MHAWithinFeat -->  def __init__()')
        self.mha = nn.MultiheadAttention(2048, 8)
        self.with_pe = cfg.MHA.WITH_PE
        self.pe = PositionalEncoding(2048)
    def forward(self, x):
        N, C, H, W = x.shape
        x = x.permute(2, 3, 0, 1)  # (16, 32, 1, 2048)
        x = x.view(-1, N, C)  # (512, 1, 2048)   (1, 512, 2048)
        if self.with_pe:
            x = self.pe(x)
        x, _ = self.mha(x, x, x)  # (512, 1, 2048)
        x = x.view(H, W, N, C)  # (16, 32, N, 2048)
        x = x.permute(2, 3, 0, 1)  # (1, 2048, 16, 32)
        return x
    def get_10x_lr_params(self):
        self.logger.info('ctrl/model_panop/mha.py --> class MHAWithinFeat -->  get_10x_lr_params()')
        b = []
        b.append(self.mha.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i
    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]

class MHAWithinFeatOld(nn.Module):
    def __init__(self, cfg, inpd, outd):
        '''
        :param cfg:
        :param inpd: channel dim
        :param outd: channel dim
        '''
        super(MHAWithinFeatOld, self).__init__()
        self.cfg = cfg
        self.inpd = inpd
        self.outd = outd
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model_panop/mha.py --> class MHAWithinFeat -->  def __init__()')
        self.mha = nn.MultiheadAttention(self.outd, 8, batch_first=True)
        self.Conv2d1 = nn.Conv2d(self.inpd, self.outd, kernel_size=1, stride=1, padding=0, bias=True)
        self.batch_norm1 = nn.BatchNorm2d(self.outd)
        self.relu1 = nn.ReLU()
        self.Conv2d2 = nn.Conv2d(self.outd, self.inpd, kernel_size=1, stride=1, padding=0, bias=True)
        self.batch_norm2 = nn.BatchNorm2d(self.inpd)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        N, C, H, W = x.shape
        x = self.Conv2d1(x)  # (1, 512, 16, 32)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = x.permute(0, 2, 3, 1)  # (1, 16, 32, 512)
        x = x.view(N, -1, self.outd)  # (1, 512, 512)
        x, _ = self.mha(x, x, x)  # (1, 512, 512)
        x = x.view(1, H, W, self.outd)  # (1, 16, 32, 512)
        x = x.permute(0, 3, 1, 2)  # (1, 512, 16, 32)
        x = self.Conv2d2(x)  # (1, 2048, 16, 32)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        return x


class MHAAcrossFeatOld(nn.Module):
    def __init__(self, cfg, inpd1, inpd2, outd):
        super(MHAAcrossFeatOld, self).__init__()
        self.cfg = cfg
        self.inpd1 = inpd1
        self.inpd2 = inpd2
        self.outd = outd
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model_panop/mha.py --> class MHAAcrossFeatOld -->  def __init__()')
        self.mha1 = nn.MultiheadAttention(self.outd, 8, batch_first=True)
        self.Conv2d11 = nn.Conv2d(self.inpd1, self.outd, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv2d12 = nn.Conv2d(self.outd, self.inpd1, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv2d21 = nn.Conv2d(self.inpd2, self.outd, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv2d22 = nn.Conv2d(self.outd, self.inpd2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        N1, C1, H1, W1 = x1.shape
        x1 = self.Conv2d11(x1)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = x1.view(N1, -1, self.outd)

        N2, C2, H2, W2 = x2.shape
        x2 = self.Conv2d21(x2)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = x2.view(N2, -1, self.outd)

        x1, _ = self.mha1(x1, x2, x1)
        x1 = x1.view(1, H1, W1, self.outd)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.Conv2d12(x1)


        x2, _ = self.mha2(x2, x1, x2)
        x2 = x2.view(1, H2, W2, self.outd)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.Conv2d22(x2)

        return x1, x2


# def positionalencoding2d(d_model, height, width):
#     """
#     :param d_model: dimension of the model
#     :param height: height of the positions
#     :param width: width of the positions
#     :return: d_model*height*width position matrix
#     """
#     if d_model % 4 != 0:
#         raise ValueError("Cannot use sin/cos positional encoding with "
#                          "odd dimension (got dim={:d})".format(d_model))
#     pe = torch.zeros(d_model, height, width)
#     # Each dimension use half of d_model
#     d_model = int(d_model / 2)
#     div_term = torch.exp(torch.arange(0., d_model, 2) *
#                          -(math.log(10000.0) / d_model))
#     pos_w = torch.arange(0., width).unsqueeze(1)
#     pos_h = torch.arange(0., height).unsqueeze(1)
#     pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
#     pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
#     return pe


# def positionalencoding1d(d_model, length):
#     """
#     :param d_model: dimension of the model
#     :param length: length of positions
#     :return: length*d_model position matrix
#     """
#     if d_model % 2 != 0:
#         raise ValueError("Cannot use sin/cos positional encoding with "
#                          "odd dim (got dim={:d})".format(d_model))
#     pe = torch.zeros(length, d_model)
#     position = torch.arange(0, length).unsqueeze(1)
#     div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
#                          -(math.log(10000.0) / d_model)))
#     pe[:, 0::2] = torch.sin(position.float() * div_term)
#     pe[:, 1::2] = torch.cos(position.float() * div_term)#
#     return pe


# ax = ax.view(H, W, N, C)
# aw_x = aw_x.sum(dim=self.sum_dim).view(H, W, N, 1) # TODO: sum will generate a H x W mask with all 1, incorrect
# ay = ay.view(H, W, N, C)
# aw_y = aw_y.sum(dim=self.sum_dim).view(H, W, N, 1)
# x = x + x * aw_y
# y = y + y * aw_x
# ax = ax + ax * aw_y
# ay = ay + ay * aw_x