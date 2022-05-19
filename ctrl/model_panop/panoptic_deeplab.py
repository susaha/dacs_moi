import torch
import torch.nn as nn
from ctrl.model_panop.backbone import resnet101
from collections import OrderedDict
from torch.nn import functional as F
from ctrl.model_panop.single_panoptic_deeplab_decoder import SinglePanopticDeepLabDecoder
from ctrl.model_panop.single_panoptic_deeplab_head import SinglePanopticDeepLabHead
import logging
from ctrl.model.aux_encoder import AuxEncoder
from ctrl.model.decoder_single_conv2d import DecSingleConv

class PanopticDeepLab(nn.Module):
    def __init__(self, cfg):
        super(PanopticDeepLab, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model_panop/panoptic_deeplab.py --> class PanopticDeepLab -->  class PanopticDeepLab() : def __init__()')
        self.cfg = cfg
        in_channels = cfg.MODEL.DECODER.IN_CHANNELS
        feature_key = cfg.MODEL.DECODER.FEATURE_KEY
        low_level_channels = tuple(cfg.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_CHANNELS)
        low_level_key = tuple(cfg.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_KEY)
        low_level_channels_project = tuple(cfg.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_CHANNELS_PROJECT)
        decoder_channels =  cfg.MODEL.DECODER.DECODER_CHANNELS
        atrous_rates = tuple(cfg.MODEL.DECODER.ATROUS_RATES)
        num_classes = cfg.NUM_CLASSES

        self.backbone = resnet101(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, replace_stride_with_dilation=(False, False, False))

        # Build semantic decoder
        self.semantic_decoder = SinglePanopticDeepLabDecoder(in_channels, feature_key, low_level_channels, low_level_key,
                                                             low_level_channels_project, decoder_channels,
                                                             atrous_rates, depth_fusion_type=self.cfg.PANOPTIC_DEEPLAB_DEPTH_FUSION_TYPE)
        self.semantic_head = SinglePanopticDeepLabHead(decoder_channels, decoder_channels, [num_classes], ['semantic'])
        self.semantic_loss_weight = cfg.LOSS.SEMANTIC.WEIGHT

        # Build instance decoder
        if True: # self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            # self.instance_decoder = None
            # self.instance_head = None
            self.center_loss_weight = cfg.LOSS.CENTER.WEIGHT
            self.offset_loss_weight = cfg.LOSS.OFFSET.WEIGHT
            low_level_channels_project_instance = tuple(cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.LOW_LEVEL_CHANNELS_PROJECT)
            decoder_channels_instance = cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.DECODER_CHANNELS
            aspp_channels_instance = cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.ASPP_CHANNELS
            head_channels = cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.HEAD_CHANNELS
            num_classes_instance = tuple(cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.NUM_CLASSES)
            class_key_instance = tuple(cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.CLASS_KEY)
            instance_decoder_kwargs = dict(
                in_channels=in_channels,
                feature_key=feature_key,
                low_level_channels=low_level_channels,
                low_level_key=low_level_key,
                low_level_channels_project=low_level_channels_project_instance,
                decoder_channels=decoder_channels_instance,
                atrous_rates=atrous_rates,
                aspp_channels=aspp_channels_instance,
            )
            self.instance_decoder = SinglePanopticDeepLabDecoder(**instance_decoder_kwargs)
            instance_head_kwargs = dict(
                decoder_channels=decoder_channels_instance,
                head_channels=head_channels,
                num_classes=num_classes_instance,
                class_key=class_key_instance,
            )
            self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            # Build depth decoder
            self.depth_loss_weight = cfg.LOSS.DEPTH.WEIGHT
            low_level_channels_project_depth = tuple(cfg.MODEL.PANOPTIC_DEEPLAB.DEPTH.LOW_LEVEL_CHANNELS_PROJECT)
            decoder_channels_depth = cfg.MODEL.PANOPTIC_DEEPLAB.DEPTH.DECODER_CHANNELS
            aspp_channels_depth = cfg.MODEL.PANOPTIC_DEEPLAB.DEPTH.ASPP_CHANNELS
            head_channels_depth = cfg.MODEL.PANOPTIC_DEEPLAB.DEPTH.HEAD_CHANNELS
            num_classes_depth = cfg.MODEL.PANOPTIC_DEEPLAB.DEPTH.NUM_CLASSES
            class_key_depth = cfg.MODEL.PANOPTIC_DEEPLAB.DEPTH.CLASS_KEY
            depth_decoder_kwargs = dict(
                in_channels=in_channels,
                feature_key=feature_key,
                low_level_channels=low_level_channels,
                low_level_key=low_level_key,
                low_level_channels_project=low_level_channels_project_depth,
                decoder_channels=decoder_channels_depth,
                atrous_rates=atrous_rates,
                aspp_channels=aspp_channels_depth,
            )
            self.depth_decoder = SinglePanopticDeepLabDecoder(**depth_decoder_kwargs)
            depth_head_kwargs = dict(
                decoder_channels=decoder_channels_depth,
                head_channels=head_channels_depth,
                num_classes=num_classes_depth,
                class_key=class_key_depth,
            )
            self.depth_head = SinglePanopticDeepLabHead(**depth_head_kwargs)

        if self.cfg.INCLUDE_DADA_AUXBLOCK:
            self.dada_aux_encoder = AuxEncoder()
            self.dada_aux_decoder = DecSingleConv(inpdim=128, outdim=2048)

        # Initialize parameters.
        self._init_params(self.semantic_decoder)
        self._init_params(self.semantic_head)
        # if True: # self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
        self._init_params(self.instance_decoder)
        self._init_params(self.instance_head)
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self._init_params(self.depth_decoder)
            self._init_params(self.depth_head)
        if self.cfg.INCLUDE_DADA_AUXBLOCK:
            self._init_params(self.dada_aux_encoder)
            self._init_params(self.dada_aux_decoder)


    # panoptic deep lab original - Use this for Oracel models
    # def forward(self, x):
    #     # contract: features is a dict of tensors
    #     features = self.backbone(x)
    #     pred = OrderedDict()
    #     # Semantic branch
    #     semantic = self.semantic_decoder(features)
    #     semantic = self.semantic_head(semantic)
    #     for key in semantic.keys():
    #         pred[key] = semantic[key]
    #     # Instance branch
    #     if True: # self.instance_decoder is not None:
    #         instance = self.instance_decoder(features)
    #         instance = self.instance_head(instance)
    #         for key in instance.keys():
    #             pred[key] = instance[key]
    #     if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
    #         depth = self.depth_decoder(features)
    #         depth = self.depth_head(depth)
    #         for key in depth.keys():
    #             pred[key] = depth[key]
    #     return pred

    # proposed dada aux block and depth fusion v2
    # def forward_depth_fusion_v2(self, x):
    #     features = self.backbone(x)
    #     # pass the res5 feature [2,2048,15,15] through the dada aux encoder and decoder
    #     if self.cfg.INCLUDE_DADA_AUXBLOCK:
    #         enc_out = self.dada_aux_encoder(features['res5'])
    #         dec_out = self.dada_aux_decoder(enc_out)
    #         features['res5'] = features['res5'] * dec_out # this is DADA feature fusion
    #     pred = OrderedDict()
    #     # Depth branch
    #     depth = self.depth_decoder(features)
    #     depth = self.depth_head(depth)
    #     for key in depth.keys():
    #         pred[key] = depth[key]
    #
    #     # Semantic branch
    #     semantic = self.semantic_decoder(features, depth['depth'])
    #     semantic = self.semantic_head(semantic)
    #     for key in semantic.keys():
    #         pred[key] = semantic[key]
    #     # Instance branch
    #     instance = self.instance_decoder(features)
    #     instance = self.instance_head(instance)
    #     for key in instance.keys():
    #         pred[key] = instance[key]
    #     return pred

    # proposed dada aux block and depth fusion v1
    def forward(self, x):
        if self.cfg.PANOPTIC_DEEPLAB_DEPTH_FUSION_TYPE == 'V1':
            features = self.backbone(x)
            # pass the res5 feature [2,2048,15,15] through the dada aux encoder and decoder
            if self.cfg.INCLUDE_DADA_AUXBLOCK:
                enc_out = self.dada_aux_encoder(features['res5'])
                dec_out = self.dada_aux_decoder(enc_out)
                features['res5'] = features['res5'] * dec_out  # this is DADA feature fusion
            pred = OrderedDict()
            # Depth branch
            depth = self.depth_decoder(features)
            depth = self.depth_head(depth)
            for key in depth.keys():
                pred[key] = depth[key]

            # do the proposed depth fusion
            if self.cfg.INCLUDE_DEPTH_FUSION:
                for k in features.keys():
                    features[k] = features[k] * F.interpolate(depth['depth'], size=features[k].size()[2:], mode='bilinear', align_corners=True)
            # Semantic branch
            semantic = self.semantic_decoder(features)
            semantic = self.semantic_head(semantic)
            for key in semantic.keys():
                pred[key] = semantic[key]
            # Instance branch
            instance = self.instance_decoder(features)
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]
            return pred

        elif self.cfg.PANOPTIC_DEEPLAB_DEPTH_FUSION_TYPE == 'V2':
            features = self.backbone(x)
            # pass the res5 feature [2,2048,15,15] through the dada aux encoder and decoder
            if self.cfg.INCLUDE_DADA_AUXBLOCK:
                enc_out = self.dada_aux_encoder(features['res5'])
                dec_out = self.dada_aux_decoder(enc_out)
                features['res5'] = features['res5'] * dec_out # this is DADA feature fusion
            pred = OrderedDict()
            # Depth branch
            depth = self.depth_decoder(features)
            depth = self.depth_head(depth)
            for key in depth.keys():
                pred[key] = depth[key]

            # Semantic branch
            semantic = self.semantic_decoder(features, depth['depth'])
            semantic = self.semantic_head(semantic)
            for key in semantic.keys():
                pred[key] = semantic[key]
            # Instance branch
            instance = self.instance_decoder(features)
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]
            return pred



    def _init_params(self, block):
        # Backbone is already initialized (either from pre-trained checkpoint or random init).
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_image_pooling(self, pool_size):
        self.semantic_decoder.set_image_pooling(pool_size)
        if True: # self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            self.instance_decoder.set_image_pooling(pool_size)
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self.depth_decoder.set_image_pooling(pool_size)

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

    # def compute_loss(self, results, targets, semantic_loss, center_loss, offset_loss, depth_loss, device):
    #     if 'semantic_weights' in targets.keys() and not self.cfg.LOSS.SEMANTIC.NAME=='dada_sem_loss':
    #         loss_semantic = semantic_loss(results['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights']) * self.semantic_loss_weight
    #         # loss_semantic = semantic_loss(results['semantic'], targets['semantic_instance'], semantic_weights=targets['semantic_weights']) * self.semantic_loss_weight
    #     else:
    #         loss_semantic = semantic_loss(results['semantic'], targets['semantic']) * self.semantic_loss_weight
    #         # loss_semantic = semantic_loss(results['semantic'], targets['semantic_instance']) * self.semantic_loss_weight
    #     loss_center = None
    #     loss_offset = None
    #     loss_depth = None
    #     if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
    #         # Pixel-wise loss weight
    #         center_loss_weights = targets['center_weights'][:, None, :, :].expand_as(results['center'])
    #         loss_center = center_loss(results['center'], targets['center']) * center_loss_weights
    #         # safe division
    #         if center_loss_weights.sum() > 0:
    #             loss_center = loss_center.sum() / center_loss_weights.sum() * self.center_loss_weight
    #         else:
    #             loss_center = loss_center.sum() * 0
    #         # Pixel-wise loss weight
    #         offset_loss_weights = targets['offset_weights'][:, None, :, :].expand_as(results['offset'])
    #         loss_offset = offset_loss(results['offset'], targets['offset']) * offset_loss_weights
    #         # safe division
    #         if offset_loss_weights.sum() > 0:
    #             loss_offset = loss_offset.sum() / offset_loss_weights.sum() * self.offset_loss_weight
    #         else:
    #             loss_offset = loss_offset.sum() * 0
    #     if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
    #         loss_depth = depth_loss(results['depth'], targets['depth']) * self.depth_loss_weight
    #     return loss_semantic, loss_center, loss_offset, loss_depth

