import torch.nn as nn
import logging



class AuxEncoder(nn.Module):
    def __init__(self):
        super(AuxEncoder, self).__init__()

        self.logger = logging.getLogger(__name__)


        self.logger.info('ctrl/model/aux_encoder.py --> class AuxEncoder -->  __init__()')
        self.enc1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.enc1(x)
        x = self.relu(x)
        x = self.enc2(x)
        x = self.relu(x)
        x = self.enc3(x)
        return x

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model/aux_encoder.py --> class AuxEncoder --> get_10x_lr_params()')
        b = []
        b.append(self.enc1.parameters())
        b.append(self.enc2.parameters())
        b.append(self.enc3.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


