import torch.nn as nn
from ctrl.model.resnet_backbone import ResnetBackbone, Bottleneck
from ctrl.model.decoder import DecoderAuxBlock
from ctrl.model.decoder_final_classifer import DecFinalClassifier
from ctrl.model.dada_depth_head import DADADetphHead
from collections import OrderedDict
from ctrl.model_panop.backbone import resnet101
import torch
import torch.nn as nn
from ctrl.model_panop.backbone import resnet101
from collections import OrderedDict
from torch.nn import functional as F
from ctrl.model_panop.single_panoptic_deeplab_decoder import SinglePanopticDeepLabDecoder
from ctrl.model_panop.single_panoptic_deeplab_head import SinglePanopticDeepLabHead
import logging
from ctrl.model_panop.mha import MHAWithinFeat
import logging

class DADAModel(nn.Module):
    def __init__(self, cfg):
        super(DADAModel, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model/mtl_aux_block.py --> class DADAModel -->  def __init__()')
        self.cfg = cfg
        # TODO: train two models with the following two backbone scripts

        if self.cfg.DADA_MODEL_BACKBONE == 'ctrl':
            self.backbone = ResnetBackbone(Bottleneck, [3, 4, 23, 3])   # ctrl backbone script
        # elif self.cfg.DADA_MODEL_BACKBONE == 'pdl':
        #     # self.backbone = resnet101(pretrained=cfg.MODEL.BACKBONE.PRETRAINED, replace_stride_with_dilation=(False, False, False)) # panoptic deeplab backbone script
        #     raise NotImplementedError('dada and panoptic deeplab backbones are different!!!')

        self.decoder = DecoderAuxBlock(inpdim=128, outdim=2048)
        self.semantic_head = DecFinalClassifier(cfg.NUM_CLASSES, 2048)
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self.depth_head = DADADetphHead()

        if self.cfg.MHA.ACTIVATE_MHA and self.cfg.MHA.TYPE == 'BACKBONE_FEAT':
            self.mha_within_feat = MHAWithinFeat(self.cfg, self.cfg.MHA.INP_DIM1, self.cfg.MHA.OUT_DIM)

        # Initialize parameters.
        self._init_params(self.decoder)
        self._init_params(self.semantic_head)
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self._init_params(self.depth_head)
        if self.cfg.MHA.ACTIVATE_MHA and self.cfg.MHA.TYPE == 'BACKBONE_FEAT':
            self._init_params(self.mha_within_feat)

        self.semantic_loss_weight = cfg.LOSS.SEMANTIC.WEIGHT
        self.center_loss_weight = cfg.LOSS.CENTER.WEIGHT
        self.offset_loss_weight = cfg.LOSS.OFFSET.WEIGHT
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self.depth_loss_weight = cfg.LOSS.DEPTH.WEIGHT

    def forward(self, x):
        pred = OrderedDict()
        x = self.backbone(x)

        if self.cfg.MHA.ACTIVATE_MHA and self.cfg.MHA.TYPE == 'BACKBONE_FEAT':
            x = self.mha_within_feat(x)

        x4_dec3, x4_dec4 = self.decoder(x)
        depth_pred = None
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            depth_pred = self.depth_head(x4_dec3)
        x = x * x4_dec4
        semantic_pred = self.semantic_head(x)
        pred['semantic'] = semantic_pred
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            pred['depth'] = depth_pred
        return pred


    # def forward_ori(self, x):
    #     pred = OrderedDict()
    #     x4 = self.backbone(x)
    #     x4_dec3, x4_dec4 = self.decoder(x4)
    #     depth_pred = self.depth_head(x4_dec3)
    #     x4 = x4 * x4_dec4
    #     semantic_pred = self.semantic_head(x4)
    #     pred['semantic'] = semantic_pred
    #     pred['depth'] = depth_pred
    #     return pred


    def _init_params(self, block):
        # Backbone is already initialized (either from pre-trained checkpoint or random init).
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        if self.cfg.USE_DEPTH:
            loss_depth = depth_loss(results['depth'], targets['depth']) * self.depth_loss_weight

        return loss_semantic, loss_center, loss_offset, loss_depth