import torch.nn as nn
import torch
import logging


class CenterHead(nn.Module):
    def __init__(self, dada_style=False):
        super(CenterHead, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.logger.info('ctrl/model/center_head.py --> class CenterHead -->  __init__()')
        self.dada_style = dada_style
        if not dada_style:
            self.enc = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        if not self.dada_style:
            x = self.enc(x)
        else:
            x = torch.mean(x, dim=1, keepdim=True)
        return x

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model/center_head.py --> class CenterHead --> get_10x_lr_params()')
        b = []
        b.append(self.enc.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


