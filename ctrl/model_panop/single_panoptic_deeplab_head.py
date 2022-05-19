from functools import partial
from torch import nn
from ctrl.model_panop.decoder.conv_module import stacked_conv
from collections import OrderedDict
import logging

class SinglePanopticDeepLabHead(nn.Module):
    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super(SinglePanopticDeepLabHead, self).__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2, conv_type='depthwise_separable_conv')

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model_panop/single_panoptic_deeplab_head.py --> class SinglePanopticDeepLabHead() : __init__()')
        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                fuse_conv(
                    decoder_channels,
                    head_channels,
                ),
                nn.Conv2d(head_channels, num_classes[i], 1)
            )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model_panop/single_panoptic_deeplab_head.py --> get_10x_lr_params()')
        b = []
        b.append(self.classifier.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

        # for i in range(len(b)):
        #     for j in b[i].modules():
        #         for k in j.parameters():
        #             if k.requires_grad:
        #                 yield k

    def optim_parameters(self, lr):
        return [{'params': self.get_10x_lr_params(), 'lr': 10 * lr}]