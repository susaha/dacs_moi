import torch.nn as nn
import logging



class DecoderAuxBlock(nn.Module):
    def __init__(self, inpdim=128, outdim=2048):
        super(DecoderAuxBlock, self).__init__()

        self.logger = logging.getLogger(__name__)


        self.logger.info('ctrl/model/decoder.py --> class DecoderAuxBlock -->  __init__()')
        self.dec1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.dec2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.dec3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.dec4 = nn.Conv2d(inpdim, outdim, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x4):
        x4_dec = self.dec1(x4)
        x4_dec = self.relu(x4_dec)
        x4_dec = self.dec2(x4_dec)
        x4_dec = self.relu(x4_dec)
        x4_dec3 = self.dec3(x4_dec)
        x4_dec4 = self.dec4(x4_dec3)
        x4_dec4 = self.relu(x4_dec4)
        return x4_dec3, x4_dec4

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model/decoder.py  --> get_10x_lr_params()')
        b = []
        b.append(self.dec1.parameters())
        b.append(self.dec2.parameters())
        b.append(self.dec3.parameters())
        b.append(self.dec4.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


