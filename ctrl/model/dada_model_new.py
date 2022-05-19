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

        self.logger = logging.getLogger(__name__)

        self.logger.info('ctrl/model/dada_model_new.py --> class DADAModel -->  def __init__()')
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
            self.aux_enc_inst = AuxEncoder()
            self.center_head = CenterHead(self.cfg.TRAIN.CENTER_HEAD_DADA_STYLE)
            if self.cfg.TRAIN.TRAIN_OFFSET_HEAD:
                self.offset_head = OffsetHead()
        # common single conv2d decoder
        if self.cfg.TRAIN.TRAIN_DEPTH_INST_TOGETHER and not self.cfg.MHA_DADA.ACTIVATE_MHA:
            if self.cfg.TRAIN.DEPTH_INST_FEAT_FUSION_TYPE_WHEHN_NO_MHA == 'cat':
                self.dec_sing_conv = DecSingleConv(inpdim=256, outdim=2048)  # COMMON DECODER
            else:
                self.dec_sing_conv = DecSingleConv(inpdim=128, outdim=2048)  # COMMON DECODER
        elif self.cfg.TRAIN.TRAIN_DEPTH_INST_TOGETHER and self.cfg.MHA_DADA.ACTIVATE_MHA:
            if self.cfg.MHA_DADA.MODE == 0 or self.cfg.MHA_DADA.MODE == 1:
                self.dec_sing_conv = DecSingleConv(inpdim=128, outdim=2048)  # COMMON DECODER
            if self.cfg.MHA_DADA.MODE == 2 or self.cfg.MHA_DADA.MODE == 3 or self.cfg.MHA_DADA.MODE == 4:
                self.dec_sing_conv = DecSingleConv(inpdim=256, outdim=2048)  # COMMON DECODER
        else:
            self.dec_sing_conv = DecSingleConv(inpdim=128, outdim=2048)  # COMMON DECODER
        # semantic deeplabv2 classifier
        self.semantic_head = DecFinalClassifier(cfg.NUM_CLASSES, 2048)
        # MHA
        if self.cfg.MHA_DADA.ACTIVATE_MHA:
            embed_dim = self.cfg.MHA_DADA.EMBED_DIM
            num_heads = self.cfg.MHA_DADA.NUM_HEADS
            mha_mode = self.cfg.MHA_DADA.MODE
            pos_enc = self.cfg.MHA_DADA.POS_ENCODING
            sum_dim = self.cfg.MHA_DADA.SUM_DIM
            self.mha = MHAAcrossFeat(embed_dim=embed_dim,num_heads=num_heads, mode=mha_mode, is_pe=pos_enc, sum_dim=sum_dim)
            self.avg_pool_src = nn.AdaptiveAvgPool2d(self.cfg.MHA_DADA.AVG_POOL_DIM_SRC)
            self.avg_pool_tar = nn.AdaptiveAvgPool2d(self.cfg.MHA_DADA.AVG_POOL_DIM_TAR)
            self.upsample_shape_src = self.cfg.MHA_DADA.UPSAMPLE_SHAPE_SRC
            self.upsample_shape_tar = self.cfg.MHA_DADA.UPSAMPLE_SHAPE_TAR

        if self.cfg.ACTIVATE_DANDA_MEMORY_MODULE:
            self.memory_weights = nn.Parameter(torch.zeros(cfg.DANDA_MEMORY_MODULE_NUM_FEATURES, cfg.DANDA_MEMORY_MODULE_FEAT_DIM))

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
        if self.cfg.MHA_DADA.ACTIVATE_MHA:
            self._init_params(self.mha)
        if self.cfg.ACTIVATE_DANDA_MEMORY_MODULE:
            nn.init.normal_(self.memory_weights, std=0.001)

        # Losses
        self.semantic_loss_weight = cfg.LOSS.SEMANTIC.WEIGHT
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            self.center_loss_weight = cfg.LOSS.CENTER.WEIGHT
            self.offset_loss_weight = cfg.LOSS.OFFSET.WEIGHT
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            self.depth_loss_weight = cfg.LOSS.DEPTH.WEIGHT


    def get_memory_block_paprams(self, lr):
        return [{'params': self.memory_weights, 'lr': 10 * lr}]
        # return [{'params': self.memory_weights.parameters(), 'lr': 10 * lr}]

    def forward(self, x, domain_label=0):
        pred = OrderedDict()
        x = self.backbone(x)

        depth_enc_out = None
        inst_enc_out = None
        enc_out = None
        if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH: # TRUE
            depth_enc_out = self.aux_enc_depth(x) # TODO: [128, H' x W'] pass this to MHA or local atten block
            pred_depth = self.depth_head(depth_enc_out)
            pred['depth'] = pred_depth
        if self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            inst_enc_out = self.aux_enc_inst(x) # TODO: [128, H' x W'] pass this to MHA or local atten block
            pred_center = self.center_head(inst_enc_out)
            pred['center'] = pred_center
            if self.cfg.TRAIN.TRAIN_OFFSET_HEAD:
                pred_offset = self.offset_head(inst_enc_out)
                pred['offset'] = pred_offset

        if self.cfg.TRAIN.TRAIN_DEPTH_INST_TOGETHER and not self.cfg.MHA_DADA.ACTIVATE_MHA:
            if self.cfg.TRAIN.DEPTH_INST_FEAT_FUSION_TYPE_WHEHN_NO_MHA == 'cat':
                enc_out = torch.cat((depth_enc_out, inst_enc_out), dim=1) # [256, H', W']
            elif self.cfg.TRAIN.DEPTH_INST_FEAT_FUSION_TYPE_WHEHN_NO_MHA == 'mul':
                enc_out = depth_enc_out * inst_enc_out
            elif self.cfg.TRAIN.DEPTH_INST_FEAT_FUSION_TYPE_WHEHN_NO_MHA == 'mean':
                enc_out = (depth_enc_out + inst_enc_out) / 2.0

        elif self.cfg.TRAIN.TRAIN_DEPTH_INST_TOGETHER and self.cfg.MHA_DADA.ACTIVATE_MHA:
            # downsample the spatial dim for MHA
            depth_enc_out = self.adap_avg_pool2d(depth_enc_out, domain_label=domain_label)
            inst_enc_out = self.adap_avg_pool2d(inst_enc_out, domain_label=domain_label)
            enc_out = self.mha(depth_enc_out, inst_enc_out)
            # upsample the spatial dim again
            if domain_label == 0:
                self.upsample_shape = self.upsample_shape_src
            else:
                self.upsample_shape = self.upsample_shape_tar
            enc_out = F.interpolate(enc_out, size=self.upsample_shape, mode='bilinear', align_corners=True)

        # this condition is true if you are training the original dada model or
        # or your new dada model with only instance encoder (not the depth enc)
        elif not self.cfg.TRAIN.TRAIN_DEPTH_INST_TOGETHER:
            if self.cfg.TRAIN.TRAIN_DEPTH_BRANCH and not self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
                enc_out = depth_enc_out # [128, H', W']
            elif not self.cfg.TRAIN.TRAIN_DEPTH_BRANCH and self.cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
                enc_out = inst_enc_out  # [128, H', W']

        dec_out = self.dec_sing_conv(enc_out) # dec_out: [2048, H', W']
        dec_out = x * dec_out
        pred_sem = self.semantic_head(dec_out)
        pred['semantic'] = pred_sem

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