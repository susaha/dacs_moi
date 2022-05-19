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
from ctrl.model.padnet.init_task_pred import InitialTaskPredictionModule, MultiTaskDistillationModule

# TASKNAMES = ["S", "D", "I"]


class DACSPanopPadNet(nn.Module):
    '''

    '''
    def __init__(self, cfg):
        super(DACSPanopPadNet, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.logger.info('ctrl/dacs_old/model/dacs_panop_model_v2.py --> class DACSPanopPadNet -->  def __init__()')
        self.cfg = cfg
        self.backbone = ResnetBackbone(Bottleneck, [3, 4, 23, 3])   # ctrl backbone script
        if self.cfg.INCLUDE_DADA_AUXBLOCK:
            self.dada_aux_encoder = AuxEncoder()
            self.dada_aux_decoder = DecSingleConv(inpdim=128, outdim=2048)

        from ctrl.model.padnet.layers import get_task_dict
        tn_dict = get_task_dict()
        self.tasks = []
        for tn in self.cfg.TASKNAMES:
            self.tasks.append(tn_dict[tn])

        # self.tasks = self.cfg.TASKNAMES
        self.auxilary_tasks = self.tasks
        self.channels = 2048
        self.NUM_OUTPUT = {"S": 16, "D": 1, "D_src": 1, "I": 128, "C": 1, "O": 2}

        # Task-specific heads for initial prediction
        self.initial_task_prediction_heads = InitialTaskPredictionModule(self.auxilary_tasks, self.channels, NUM_OUTPUT=self.NUM_OUTPUT) # TODO: add the param to the optimizer
        # Multi-modal distillation
        self.multi_modal_distillation = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, 256)   # TODO: add the param to the optimizer

        heads = {}
        for task in self.tasks + ["D_src"]:
            heads[task] = self._make_pred_layer(DecFinalClassifier, input_dim=256, output_dim=self.NUM_OUTPUT[task])
        self.heads = nn.ModuleDict(heads)

        self.center_sub_head = CenterHead(self.cfg.TRAIN.CENTER_HEAD_DADA_STYLE)
        self.offset_sub_head = OffsetHead()


        # Initialize parameters
        if self.cfg.INCLUDE_DADA_AUXBLOCK:
            self._init_params(self.dada_aux_encoder)
            self._init_params(self.dada_aux_decoder)

        self._init_params(self.initial_task_prediction_heads)
        self._init_params(self.multi_modal_distillation)

        for task in self.tasks + ["D_src"]:
            self._init_params(self.heads[task])

        self._init_params(self.center_sub_head)
        self._init_params(self.offset_sub_head)


    def _make_pred_layer(self, block, input_dim=256, output_dim=None):
        return block(output_dim, input_dim)


    def forward(self, x, domain_label=0):
        out = OrderedDict()
        x = self.backbone(x)

        if self.cfg.INCLUDE_DADA_AUXBLOCK:
            enc_out = self.dada_aux_encoder(x)
            dec_out = self.dada_aux_decoder(enc_out)
            x = x * dec_out

        # Initial predictions for every task including auxilary tasks
        # x['features_S'], x['features_D'], x['features_I'], x['S'], x['D'], x['C'], x['O']
        # out['inital_S'], out['inital_D'], out['inital_C'], out['inital_O']
        x = self.initial_task_prediction_heads(x)
        for task in self.auxilary_tasks:
            if task == 'I':
                out['initial_%s' % ('C')] = x['C']
                out['initial_%s' % ('O')] = x['O']
            else:
                out['initial_%s' % (task)] = x[task]
        out["initial_D_src"] = x["D_src"]

        # Refine features through multi-modal distillation
        # x['S'], x['D'], X['I'] --> attention based task features
        x = self.multi_modal_distillation(x)

        # Make final prediction with task-specific heads
        for task in self.tasks:
            if task == "S":
                out['semantic'] = self.heads[task](x[task])
            if task == "D":
                out["depth"] = self.heads[task](x[task])
                out["D_src"] = self.heads["D_src"](x[task])
            if task == "I":
                inst_out = self.heads[task](x[task])
                out['center'] = self.center_sub_head(inst_out)
                out['offset'] = self.offset_sub_head(inst_out)

        # out['semantic'] = self.semantic_head(x['S'])
        # out['depth'] = self.depth_head(x['D'])
        # inst_x = self.instance_head(x['I'])
        # out['center'] = self.center_sub_head(inst_x)
        # out['offset'] = self.offset_sub_head(inst_x)

        return out


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
        # odict_keys(['initial_S', 'initial_D', 'initial_C', 'initial_O', 'semantic', 'depth', 'center', 'offset'])
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            if 'offset' in key or 'initial_O':
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