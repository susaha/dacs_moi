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

class DADAModel(nn.Module):
    '''
    - there are mainly two way to design the net wrch
    APPRACOH-1: - initialized the backbone, single donv2d decoder,deeplabv2 clasifier with dada weights
    - init the instance encoder and heads with radnom noise
    - train all of them
    - this approach is exactly same as dada,just you replace the depth encoder with inst encoder
    - and add two additional conv for center and offset
    - if you want for center you don't need to add a conv and  you can do a mean pool as done for depth head in dada

    APPROACH-2: if you want to keep the depth encoder them you need to find a way of fusing the
    conv feature map output by the depth and instance encoders
    these two feature maps are of dim 128 x H' x W'
    the simplest way is to concat these two features to 256 x H' x W'
    and then pass this to the common single conv decoder
    under this you can  have the following sub approaches:
    - train the entire network from scracth for depth instance and semantic
    - init the network with dada weights and then freeze the depth encoder and just train the instance encoder
    - train the depth and instance encoder from scratch
    - add a attention block between depth and instance and then either train from scratch or freeze depth and train only instance
    -
    '''
    def __init__(self, cfg):
        super(DADAModel, self).__init__()
        print('ctrl/model/dada_model_new_v2.py --> class DADAModel -->  def __init__()')
        self.cfg = cfg
        # Backbone
        if self.cfg.DADA_MODEL_BACKBONE == 'ctrl':
            self.backbone = ResnetBackbone(Bottleneck, [3, 4, 23, 3])   # ctrl backbone script
        # Depth encoder
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self.aux_enc_depth = AuxEncoder()
            # DADADetphHead has no param, this performs mean pooling over channel dim
            # of shape 128 x H x W featuremap and map to 1 x H x W
            self.depth_head = DADADetphHead()
        # Instance encoder
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            # self.aux_enc_inst = AuxEncoder()
            self.aux_enc_inst = DecFinalClassifier(128, 2048)
            self.center_head = CenterHead(self.cfg.TRAIN.CENTER_HEAD_DADA_STYLE)
            if self.cfg.TRAIN.TRAIN_OFFSET_HEAD:
                self.offset_head = OffsetHead()

        # common single conv2d decoder
        self.dec_sing_conv = DecSingleConv(inpdim=128, outdim=2048)  # COMMON DECODER

        # semantic deeplabv2 classifier
        self.semantic_head = DecFinalClassifier(cfg.NUM_CLASSES, 2048)

        # Initialize parameters
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self._init_params(self.aux_enc_depth)
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            self._init_params(self.aux_enc_inst)
            if not self.cfg.TRAIN.CENTER_HEAD_DADA_STYLE:
                self._init_params(self.center_head)
            if self.cfg.TRAIN.TRAIN_OFFSET_HEAD:
                self._init_params(self.offset_head)
        self._init_params(self.dec_sing_conv)
        self._init_params(self.semantic_head)

        # Losses
        self.semantic_loss_weight = cfg.LOSS.SEMANTIC.WEIGHT
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            self.center_loss_weight = cfg.LOSS.CENTER.WEIGHT
            self.offset_loss_weight = cfg.LOSS.OFFSET.WEIGHT
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self.depth_loss_weight = cfg.LOSS.DEPTH.WEIGHT

    def forward(self, x, domain_label=0):
        pred = OrderedDict()
        x = self.backbone(x)

        depth_enc_out = None
        inst_enc_out = None
        enc_out = None
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            depth_enc_out = self.aux_enc_depth(x) # TODO: [128, H' x W'] pass this to MHA or local atten block
            pred_depth = self.depth_head(depth_enc_out)
            pred['depth'] = pred_depth

        dec_out = self.dec_sing_conv(depth_enc_out)  # dec_out: [2048, H', W']
        dec_out = x * dec_out
        pred_sem = self.semantic_head(dec_out)
        pred['semantic'] = pred_sem

        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            inst_enc_out = self.aux_enc_inst(dec_out)
            pred_center = self.center_head(inst_enc_out)
            pred['center'] = pred_center
            if self.cfg.TRAIN.TRAIN_OFFSET_HEAD:
                pred_offset = self.offset_head(inst_enc_out)
                pred['offset'] = pred_offset

        return pred

    def adap_avg_pool2d(self, x, domain_label=0):
        if domain_label == 0:
            x = self.avg_pool_src(x)
        elif domain_label == 1:
            x = self.avg_pool_tar(x)
        return x


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