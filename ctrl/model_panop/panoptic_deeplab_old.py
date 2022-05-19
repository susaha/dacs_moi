import torch
import torch.nn as nn
from ctrl.model_panop.backbone import resnet101
from collections import OrderedDict
from torch.nn import functional as F
from ctrl.model_panop.single_panoptic_deeplab_decoder import SinglePanopticDeepLabDecoder
from ctrl.model_panop.single_panoptic_deeplab_head import SinglePanopticDeepLabHead
import logging
from ctrl.model_panop.mha import MHAWithinFeat, MHACrossFeat


class PanopticDeepLab(nn.Module):
    def __init__(self, cfg):
        super(PanopticDeepLab, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model_panop/panoptic_deeplab_old.py --> class PanopticDeepLab -->  class PanopticDeepLab() : def __init__()')
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
        self.semantic_decoder = SinglePanopticDeepLabDecoder(in_channels, feature_key, low_level_channels,low_level_key, low_level_channels_project, decoder_channels, atrous_rates)
        self.semantic_head = SinglePanopticDeepLabHead(decoder_channels, decoder_channels, [num_classes], ['semantic'])
        self.semantic_loss_weight = cfg.LOSS.SEMANTIC.WEIGHT

        # Build instance decoder
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
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


        # added for MHA stuff
        if self.cfg.MHA.ACTIVATE_MHA and self.cfg.MHA.TYPE == 'BACKBONE_FEAT':
            self.mha_within_feat = MHAWithinFeat(self.cfg, self.cfg.MHA.INP_DIM1, self.cfg.MHA.OUT_DIM)
        if self.cfg.MHA.ACTIVATE_MHA and self.cfg.MHA.TYPE == 'CROSS_TASK_SEM_DEP':
            self.mha_cross_feat = MHACrossFeat(self.cfg, self.cfg.MHA.INP_DIM2, self.cfg.MHA.INP_DIM3, self.cfg.MHA.OUT_DIM2)

        # Initialize parameters.
        self._init_params(self.semantic_decoder)
        self._init_params(self.semantic_head)
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            self._init_params(self.instance_decoder)
            self._init_params(self.instance_head)
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self._init_params(self.depth_decoder)
            self._init_params(self.depth_head)
        if self.cfg.MHA.ACTIVATE_MHA and self.cfg.MHA.TYPE == 'BACKBONE_FEAT':
            self._init_params(self.mha_within_feat)
        if self.cfg.MHA.ACTIVATE_MHA and self.cfg.MHA.TYPE == 'CROSS_TASK_SEM_DEP':
            self._init_params(self.mha_cross_feat)

    def forward(self, x):
        '''
         features: {dict: 5}
        'stem' = {Tensor: (1, 64, 128, 256)}
        'res2' = {Tensor: (1, 256, 128, 256)}
        'res3' = {Tensor: (1, 512, 64, 128)}
        'res4' = {Tensor: (1, 1024, 32, 64}
        'res5' = {Tensor: (1, 2048, 16, 32)}
        semantic = self.semantic_decoder(features)
        semantic = {Tensor:(1, 256, 128, 256)}
        semantic = self.semantic_head(semantic)
        semantic['semantic'] = {Tensor: (1, 19, 128, 256)}
        instance = self.instance_decoder(features)
        instance = {Tensor: (1, 128, 128, 256)}
        instance = self.instance_head(instance)
        '''
        # contract: features is a dict of tensors
        features = self.backbone(x)

        if self.cfg.MHA.ACTIVATE_MHA and self.cfg.MHA.TYPE == 'BACKBONE_FEAT':
            if self.cfg.MHA.LOCATION == 'BACKBONE_RES5':
                feat_res5 = self.mha_within_feat(features['res5'])
                if self.cfg.MHA.FUSION_TYPE1 == 'mul':
                    features['res5'] = features['res5'] * feat_res5
                elif self.cfg.MHA.FUSION_TYPE1 == 'add':
                    features['res5'] = features['res5'] + feat_res5

        pred = OrderedDict()
        # Semantic decoder
        semantic = self.semantic_decoder(features)
        # Instance decoder
        instance = None
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            instance = self.instance_decoder(features)
        # Depth decoder
        depth = None
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            depth = self.depth_decoder(features)
        # cross-task MHA
        if self.cfg.MHA.ACTIVATE_MHA and self.cfg.MHA.LOCATION == 'DECODER_FEAT':
            if self.cfg.MHA.TYPE == 'CROSS_TASK_SEM_DEP':
                semantic, depth = self.mha_cross_feat(semantic, depth)
        # Semanitc head
        semantic = self.semantic_head(semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]
        # Instance head
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]
        # Depth head
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            depth = self.depth_head(depth)
            for key in depth.keys():
                pred[key] = depth[key]
        return pred

    def forward_original(self, x):
        # contract: features is a dict of tensors
        features = self.backbone(x)
        pred = OrderedDict()
        # Semantic branch
        semantic = self.semantic_decoder(features)
        semantic = self.semantic_head(semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]
        # Instance branch
        if self.instance_decoder is not None:
            instance = self.instance_decoder(features)
            instance = self.instance_head(instance)
            for key in instance.keys():
                pred[key] = instance[key]
        if self.cfg.USE_DEPTH:
            depth = self.depth_decoder(features)
            depth = self.depth_head(depth)
            for key in depth.keys():
                pred[key] = depth[key]
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
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
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

