import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from ctrl.model.padnet.layers import SEBlock, SABlock
from ctrl.model.padnet.bottleneck_pad import BottleneckPad
import logging

affine_par = True


# NUM_OUTPUT = {"S": 16, "D": 1, "C": 1, "O": 2}
# TASKNAMES = ["S", "D", "I"]


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """

    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model/padnet/init_task_pred.py --> class MultiTaskDistillationModule -->  def __init__()')

        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}

        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)

    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' % (a)]) for a in self.auxilary_tasks if a != t} for t in self.tasks}
        out = {t: x['features_%s' % (t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in self.tasks}
        return out

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model/padnet/init_task_pred.py --> MultiTaskDistillationModule --> get_10x_lr_params()')
        b = []
        b.append(self.self_attention.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """
    def __init__(self, tasks, input_channels, intermediate_channels=256, NUM_OUTPUT=None):
        super(InitialTaskPredictionModule, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model/padnet/init_task_pred.py --> class InitialTaskPredictionModule -->  def __init__()')

        self.tasks = tasks
        layers = {}
        conv_out = {}

        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = BottleneckPad(input_channels, intermediate_channels // 4, downsample=downsample)
            bottleneck2 = BottleneckPad(intermediate_channels, intermediate_channels // 4, downsample=None)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            if task == 'I':
                conv_out['C'] = nn.Conv2d(intermediate_channels, NUM_OUTPUT['C'], 1)
                conv_out['O'] = nn.Conv2d(intermediate_channels, NUM_OUTPUT['O'], 1)
            else:
                conv_out[task] = nn.Conv2d(intermediate_channels, NUM_OUTPUT[task], 1)

        conv_out["D_src"] = nn.Conv2d(intermediate_channels, NUM_OUTPUT["D"], 1)

        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)

    def forward(self, x):
        out = {}
        for task in self.tasks:
            out['features_%s' % (task)] = self.layers[task](x)
            if task == 'I':
                out['C'] = self.conv_out['C'](out['features_%s' % (task)])
                out['O'] = self.conv_out['O'](out['features_%s' % (task)])
            else:
                out[task] = self.conv_out[task](out['features_%s' % (task)])
        out["D_src"] = self.conv_out["D_src"](out['features_D'])
        return out

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model/padnet/init_task_pred.py --> InitialTaskPredictionModule --> get_10x_lr_params()')
        b = []
        b.append(self.layers.parameters())
        b.append(self.conv_out.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]