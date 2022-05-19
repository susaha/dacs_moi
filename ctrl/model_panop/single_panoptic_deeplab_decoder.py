from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from ctrl.model_panop.decoder.aspp import ASPP
from ctrl.model_panop.decoder.conv_module import stacked_conv
import logging

class SinglePanopticDeepLabDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels,
                 low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, aspp_channels=None, depth_fusion_type=None):

        super(SinglePanopticDeepLabDecoder, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model_panop/single_panoptic_deeplab_decoder.py --> class SinglePanopticDeepLabDecoder() : __init__()')
        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp = ASPP(in_channels, out_channels=aspp_channels, atrous_rates=atrous_rates)
        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2, conv_type='depthwise_separable_conv')

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(low_level_channels[i], low_level_channels_project[i], 1, bias=False),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU()
                )
            )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_conv(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)
        self.depth_fusion_type = depth_fusion_type

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features, depth=None):
        if self.depth_fusion_type == 'V2': # TODO : depth fusion V2
            x = features[self.feature_key]
            x = self.aspp(x * F.interpolate(depth, size=x.size()[2:], mode='bilinear', align_corners=True))
            # build decoder
            for i in range(self.decoder_stage):
                l = features[self.low_level_key[i]]
                l = self.project[i](l * F.interpolate(depth, size=l.size()[2:], mode='bilinear', align_corners=True))
                x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
                x = torch.cat((x, l), dim=1)
                x = self.fuse[i](x)
            return x

        elif self.depth_fusion_type == 'V1' or not self.depth_fusion_type:  # original forward pass
            x = features[self.feature_key]
            x = self.aspp(x)
            # build decoder
            for i in range(self.decoder_stage):
                l = features[self.low_level_key[i]]
                l = self.project[i](l)
                x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
                x = torch.cat((x, l), dim=1)
                x = self.fuse[i](x)
            return x

    def get_10x_lr_params(self):
        self.logger.info('ctrl/model_panop/single_panoptic_deeplab_decoder.py --> get_10x_lr_params()')
        b = []
        b.append(self.aspp.parameters())
        b.append(self.project.parameters())
        b.append(self.fuse.parameters())

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





