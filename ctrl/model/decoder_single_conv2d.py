import torch.nn as nn
import logging



class DecSingleConv(nn.Module):
    def __init__(self, inpdim=128, outdim=2048):
        super(DecSingleConv, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.logger.info('ctrl/model/decoder_single_conv2d.py --> class DecSingleConv -->  __init__()')
        self.dec = nn.Conv2d(inpdim, outdim, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dec(x)
        x = self.relu(x)
        return x

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model/decoder_single_conv2d.py --> class DecSingleConv --> get_10x_lr_params()')
        b = []
        b.append(self.dec.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


