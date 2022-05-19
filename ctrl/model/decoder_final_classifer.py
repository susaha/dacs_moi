import torch.nn as nn
import logging


class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class DecFinalClassifier(nn.Module):
    def __init__(self, num_classes, inpdim=2048):
        super(DecFinalClassifier, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model/decoder_final_classifer.py --> class DecFinalClassifier -->  __init__()')
        self.layer6 = ClassifierModule(inpdim, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def forward(self, x4):
        out = self.layer6(x4)
        return out

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model/decoder_final_classifer.py --> get_10x_lr_params()')
        b = []
        b.append(self.layer6.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]