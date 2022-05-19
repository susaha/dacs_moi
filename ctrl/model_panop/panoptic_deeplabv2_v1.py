import torch.nn as nn
from ctrl.model.resnet_backbone import ResnetBackbone, Bottleneck
from ctrl.model.decoder import DecoderAuxBlock
from ctrl.model.decoder_final_classifer import DecFinalClassifier
import torch.nn as nn
from ctrl.model_panop.backbone import resnet101
from collections import OrderedDict
from torch.nn import functional as F
from ctrl.model_panop.single_panoptic_deeplab_decoder import SinglePanopticDeepLabDecoder
from ctrl.model_panop.single_panoptic_deeplab_head import SinglePanopticDeepLabHead
import logging
import torch



class PanopticDeepLab(nn.Module):
    '''
    DecoderAuxBlock: takes 2048 x H x W channel tensor and outputs 2048 x H x W tensor
    '''
    def __init__(self, cfg):
        super(PanopticDeepLab, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.logger.info('ctrl/model_panop/panoptic_deeplabv2_v1.py --> class PanopticDeepLab -->  class PanopticDeepLab() : def __init__()')
        self.cfg = cfg
        num_classes = cfg.NUM_CLASSES
        decoder_channels = cfg.MODEL.DECODER.DECODER_CHANNELS

        # ctrl backbone
        self.backbone = ResnetBackbone(Bottleneck, [3, 4, 23, 3])
        # panoptic-deeplab backbone
        # self.backbone = resnet101(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, replace_stride_with_dilation=(False, False, False))
        self.aux_block = DecoderAuxBlock(inpdim=128, outdim=2048) # try with and without this
        # this will output 256 x H x W tensor which will be passed to the SinglePanopticDeepLabHead
        self.semantic_deeplabv2_decoder = DecFinalClassifier(256, 2048)
        # panoptic-deeplab semantic_head
        self.semantic_head = SinglePanopticDeepLabHead(decoder_channels, decoder_channels, [num_classes], ['semantic'])
        self.semantic_loss_weight = cfg.LOSS.SEMANTIC.WEIGHT

        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            self.center_loss_weight = cfg.LOSS.CENTER.WEIGHT
            self.offset_loss_weight = cfg.LOSS.OFFSET.WEIGHT
            decoder_channels_instance = cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.DECODER_CHANNELS
            head_channels = cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.HEAD_CHANNELS
            num_classes_instance = tuple(cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.NUM_CLASSES)
            class_key_instance = tuple(cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.CLASS_KEY)
            # this will output 128 x H x W tensor which will be passed to the SinglePanopticDeepLabHead
            self.instance_deeplabv2_decoder = DecFinalClassifier(128, 2048)
            instance_head_kwargs = dict(
                decoder_channels=decoder_channels_instance,
                head_channels=head_channels,
                num_classes=num_classes_instance,
                class_key=class_key_instance,
            )
            # panoptic-deeplab instance head
            self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self.depth_loss_weight = cfg.LOSS.DEPTH.WEIGHT
            decoder_channels_depth = cfg.MODEL.PANOPTIC_DEEPLAB.DEPTH.DECODER_CHANNELS
            depth_head_channels = cfg.MODEL.PANOPTIC_DEEPLAB.DEPTH.HEAD_CHANNELS
            depth_head_kwargs = dict(
                decoder_channels=decoder_channels_depth,
                head_channels=depth_head_channels,
                num_classes=[1],
                class_key=['depth'],
            )
            self.depth_deeplabv2_decoder = DecFinalClassifier(128, 2048)
            self.depth_head = SinglePanopticDeepLabHead(**depth_head_kwargs)

        # Initialize parameters.
        self._init_params(self.aux_block)
        self._init_params(self.semantic_deeplabv2_decoder)
        self._init_params(self.semantic_head)
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            self._init_params(self.instance_deeplabv2_decoder)
            self._init_params(self.instance_head)
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self._init_params(self.depth_deeplabv2_decoder)
            self._init_params(self.depth_head)


    def _init_params(self, block):
        # Backbone is already initialized (either from pre-trained checkpoint or random init).
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # contract: features is a dict of tensors
        x4 = self.backbone(x)
        x4_dec3, x4_dec4 = self.aux_block(x4)
        x4 = x4 * x4_dec4

        pred = OrderedDict()
        x4_semantic = self.semantic_deeplabv2_decoder(x4)
        semantic = self.semantic_head(x4_semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]

        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            x4_instance = self.instance_deeplabv2_decoder(x4)
            instance = self.instance_head(x4_instance)
            for key in instance.keys():
                pred[key] = instance[key]

        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            x4_depth = self.depth_deeplabv2_decoder(x4)
            depth = self.depth_head(x4_depth)
            for key in depth.keys():
                pred[key] = depth[key]

        return pred

    def upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
            Args:
                pred (dict): stores all output of the segmentation model.
                input_shape (tuple): spatial resolution of the desired shape.
            Returns:
                result (OrderedDict): upsampled dictionary.
            """
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def compute_loss(self, results, targets, semantic_loss, center_loss, offset_loss, depth_loss, device):

        if 'semantic_weights' in targets.keys() and not self.cfg.LOSS.SEMANTIC.NAME=='dada_sem_loss':
            loss_semantic = semantic_loss(results['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights']) * self.semantic_loss_weight
            # loss_semantic = semantic_loss(results['semantic'], targets['semantic_instance'], semantic_weights=targets['semantic_weights']) * self.semantic_loss_weight
        else:
            loss_semantic = semantic_loss(results['semantic'], targets['semantic']) * self.semantic_loss_weight
            # loss_semantic = semantic_loss(results['semantic'], targets['semantic_instance']) * self.semantic_loss_weight

        loss_center = None
        loss_offset = None
        loss_depth = None
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            # Pixel-wise loss weight
            center_loss_weights = targets['center_weights'][:, None, :, :].expand_as(results['center'])
            loss_center = center_loss(results['center'], targets['center']) * center_loss_weights
            # safe division
            if center_loss_weights.sum() > 0:
                loss_center = loss_center.sum() / center_loss_weights.sum() * self.center_loss_weight
            else:
                loss_center = loss_center.sum() * 0
            # Pixel-wise loss weight
            offset_loss_weights = targets['offset_weights'][:, None, :, :].expand_as(results['offset'])
            loss_offset = offset_loss(results['offset'], targets['offset']) * offset_loss_weights
            # safe division
            if offset_loss_weights.sum() > 0:
                loss_offset = loss_offset.sum() / offset_loss_weights.sum() * self.offset_loss_weight
            else:
                loss_offset = loss_offset.sum() * 0

        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            loss_depth = depth_loss(results['depth'], targets['depth']) * self.depth_loss_weight

        return loss_semantic, loss_center, loss_offset, loss_depth


