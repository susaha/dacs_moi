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
from ctrl.model.aux_encoder import AuxEncoder
from ctrl.model.decoder_single_conv2d import DecSingleConv
from ctrl.model.center_head import CenterHead
from ctrl.model.offset_head import OffsetHead
from ctrl.model_panop.mha import MHAAcrossFeat
import logging
from ctrl.model.aux_encoder import AuxEncoder
from ctrl.model.decoder_single_conv2d import DecSingleConv


class DACSPanop(nn.Module):
    def __init__(self, cfg):
        super(DACSPanop, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/dacs_old/model/dacs_panop_model.py --> class DACSPanop -->  def __init__()')
        self.cfg = cfg
        # Backbone
        self.backbone = ResnetBackbone(Bottleneck, [3, 4, 23, 3])   # ctrl backbone script
        if self.cfg.INCLUDE_DADA_AUXBLOCK:
            self.dada_aux_encoder = AuxEncoder()
            self.dada_aux_decoder = DecSingleConv(inpdim=128, outdim=2048)
        # semantic deeplabv2 head
        self.semantic_head = DecFinalClassifier(cfg.NUM_CLASSES, 2048)
        # instance deeplabv2 head
        self.instance_head = DecFinalClassifier(128, 2048)
        # sub heads (one conv layer) branch out from instance head
        self.center_sub_head = CenterHead(self.cfg.TRAIN.CENTER_HEAD_DADA_STYLE)
        self.offset_sub_head = OffsetHead()
        # Depth encoder
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self.depth_head = DADADetphHead()
        # Initialize parameters
        self._init_params(self.semantic_head)
        self._init_params(self.instance_head)
        self._init_params(self.center_sub_head)
        self._init_params(self.offset_sub_head)
        if self.cfg.INCLUDE_DADA_AUXBLOCK:
            self._init_params(self.dada_aux_encoder)
            self._init_params(self.dada_aux_decoder)

    def forward(self, x, domain_label=0):
        pred = OrderedDict()
        x = self.backbone(x)
        if self.cfg.INCLUDE_DADA_AUXBLOCK:
            enc_out = self.dada_aux_encoder(x)
            if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
                pred['depth'] = self.depth_head(enc_out)
            dec_out = self.dada_aux_decoder(enc_out)
            dec_out = x * dec_out
            pred['semantic'] = self.semantic_head(dec_out)
            inst_x = self.instance_head(dec_out)
            pred['center'] = self.center_sub_head(inst_x)
            pred['offset'] = self.offset_sub_head(inst_x)
        else:
            pred['semantic'] = self.semantic_head(x)
            inst_x = self.instance_head(x)
            pred['center'] = self.center_sub_head(inst_x)
            pred['offset'] = self.offset_sub_head(inst_x)
        return pred


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

        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            loss_depth = depth_loss(results['depth'], targets['depth']) * self.depth_loss_weight

        return loss_semantic, loss_center, loss_offset, loss_depth