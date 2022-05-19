import torch.nn as nn
import logging


class OffsetHead(nn.Module):
    def __init__(self):
        super(OffsetHead, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model/offset_head.py --> class OffsetHead -->  __init__()')
        self.enc = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.enc(x)
        return x

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model/offset_head.py --> class OffsetHead --> get_10x_lr_params()')
        b = []
        b.append(self.enc.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


