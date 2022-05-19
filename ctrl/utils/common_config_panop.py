import torch
from ctrl.model.get_disc import get_discriminator
import numpy as np
from torch.utils import data
import yaml
from easydict import EasyDict as edict
from datetime import datetime
import os
import os.path as osp
from ctrl.model.mtl_aux_block import MTLAuxBlock

from ctrl.utils.panoptic_deeplab.env import seed_all_rng
from ctrl.utils.panoptic_deeplab.utils import get_loss_info_str, get_module
import logging
import torch.nn as nn
from torch.hub import load_state_dict_from_url as load_url


def convert_yaml_to_edict(exp_file):
    with open(exp_file, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = edict()
    for k, v in config.items():
        if type(v) is dict:
            v = edict(v)
        cfg[k] = v
    return cfg


def get_model(cfg, mode='train'):

    logger = logging.getLogger(__name__)
    model = None
    ema_model = None

    if cfg.MODEL_TYPE == 'cvpr2021':
        model = MTLAuxBlock(cfg.NUM_CLASSES)

    elif cfg.MODEL_TYPE == 'panop_v1':  # panoptic_deeplab original model from CVPR 2020 paper
        PanopticDeepLab = None
        if cfg.MODEL_SUB_TYPE == 'deeplabv3':
            from ctrl.model_panop.panoptic_deeplab_old import PanopticDeepLab
        elif cfg.MODEL_SUB_TYPE == 'deeplabv2':
            from ctrl.model_panop.panoptic_deeplabv2_v1 import PanopticDeepLab
        model = PanopticDeepLab(cfg)
        if cfg.PRETRAINED_WEIGHTS_FOR_TRAIN == 'DADA_IMGNET':
            if not mode == 'val':
                model_params_current = model.state_dict().copy()
                model_params_saved = torch.load(cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET)
                for i in model_params_saved:
                    i_parts = i.split(".")
                    if not i_parts[1] == "layer5":
                        model_params_current['backbone.{}'.format(".".join(i_parts[1:]))] = model_params_saved[i]
                model.load_state_dict(model_params_current)
                logger.info('ResNet Backbone is initialized with dada deeplab resent pretrained ImageNet weights')
                logger.info('from: {}'.format(cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET))
        logger.info('***')
        logger.info('cfg.MODEL_SUB_TYPE: {}'.format(cfg.MODEL_SUB_TYPE))
        logger.info('cfg.MODEL.BN_MOMENTUM: {}'.format(cfg.MODEL.BN_MOMENTUM))
        if cfg.TRAIN.FREEZE_BN:
            logger.info('cfg.TRAIN.FREEZE_BN: {}'.format(cfg.TRAIN.FREEZE_BN))
        else:
            logger.info('cfg.TRAIN.FREEZE_BN: {}'.format(cfg.TRAIN.FREEZE_BN))
        logger.info('***')
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                if cfg.TRAIN.FREEZE_BN:
                    module.requires_grad_(False)
                else:
                    module.requires_grad_(True)
                module.momentum = cfg.MODEL.BN_MOMENTUM

    # TODO: new entry
    elif cfg.MODEL_TYPE == 'dacs_old':
        from ctrl.dacs_old.model.dacs_model import DACSModel
        model = DACSModel(cfg)
        model_params_current = model.state_dict().copy()

        if cfg.WEIGHT_INITIALIZATION.DACS_COCO_MODEL and not mode == 'val':
            model_params_saved = load_url(cfg.TRAIN.DACS_IMAGENET_COCO_PRETRAINED_MODEL)
            logger.info('ResNet Backbone is initialized with dacs_old imagenet coco pretrained weights')
            logger.info('from: {}'.format(cfg.TRAIN.DACS_IMAGENET_COCO_PRETRAINED_MODEL))
            model_params_current = model.state_dict().copy()
            for name, param in model_params_current.items():
                name_parts = name.split(".")
                name2 = ".".join(name_parts[1:])
                if name2 in model_params_saved and param.size() == model_params_saved[name2].size():
                    # print('loading weights from {} to {}'.format(name, name2))
                    model_params_current[name].copy_(model_params_saved[name2])

        elif cfg.WEIGHT_INITIALIZATION.DADA_DEEPLABV2 and not mode == 'val':
            model_params_saved = torch.load(cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET)
            logger.info('ResNet Backbone is initialized with dada deeplab resent pretrained ImageNet weights')
            logger.info('from: {}'.format(cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET))
            for i in model_params_saved:
                i_parts = i.split(".")
                if not i_parts[1] == "layer5":
                    model_params_current['backbone.{}'.format(".".join(i_parts[1:]))] = model_params_saved[i]

        elif cfg.WEIGHT_INITIALIZATION.DADA_PRETRAINED and cfg.MODEL_FILE == 'dada_model_new' and not mode == 'val':
            model_params_saved = torch.load(cfg.TRAIN.DADA_PRETRAINED_MODEL_FILE_PATH)
            model_params_current = model.state_dict().copy()
            # init the resent backbone
            for i in model_params_saved['model_state_dict']:
                i_parts = i.split(".")
                if i_parts[0] == "backbone":
                    model_params_current[i] = model_params_saved['model_state_dict'][i]
            logger.info('ResNet Backbone is initialized with dada pretrained weights')

        # LOADING THE WEIGHTS
        model.load_state_dict(model_params_current)

        logger.info('cfg.MODEL_SUB_TYPE: {}'.format(cfg.MODEL_SUB_TYPE))
        logger.info('cfg.TRAIN.FREEZE_BN: {}'.format(cfg.TRAIN.FREEZE_BN))
        logger.info('***')
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                if cfg.TRAIN.FREEZE_BN:
                    module.requires_grad_(False)
                else:
                    module.requires_grad_(True)

        '''
        DACS used Mean teachers approach as in * for the teacher-student model used for genrating pseudo labels
        # EMA Moddel (exponential moving average)
        * Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
        https://arxiv.org/pdf/1703.01780.pdf
        '''
        from ctrl.dacs_old.utils.train_uda_scripts import create_ema_model
        ema_model = create_ema_model(model, cfg)

    elif cfg.MODEL_TYPE == 'dada' and cfg.MODEL_SUB_TYPE=='deeplabv2':
        '''
        dada_model_new_v2: has two parallel deeplabv2 decoder for semantic and instances, and depth encoder is same as dada
        dada_model_new: is is same as dada, the instance encoder (similar to dada depth encoder) has been added following the MTL Auxiliary block
        '''
        if cfg.MODEL_FILE == 'dada_model_new':
            from ctrl.model.dada_model_new import DADAModel
        elif cfg.MODEL_FILE == 'dada_model_new_v2':
            from ctrl.model.dada_model_new_v2 import DADAModel
        else:
            from ctrl.model.dada_model import DADAModel
        model = DADAModel(cfg)
        logger.info('***')

        if cfg.WEIGHT_INITIALIZATION.CORDA_PRETRAINED_MODEL_S2C and not mode == 'val':
            # dict_keys(['iteration', 'optimizer', 'config', 'model', 'ema_model'])
            model_params_current = model.state_dict().copy()
            model_params_saved = torch.load(cfg.TRAIN.CORDA_PRETRAINED_MODEL_S2C)['model']

            # for i in model_params_current:
            #     print(i)
            # print('--------------')
            for i in model_params_current:
                i_parts = i.split(".")
                if i_parts[0] == "backbone":
                    str1 = '{}'.format(".".join(i_parts[1:]))
                    print('{} :  {}'.format(i, str1))
                    model_params_current[i] = model_params_saved[str1]

            model.load_state_dict(model_params_current)
            logger.info('ResNet Backbone is initialized with CORDA pretrained model\'s (syhtia to cityscapes) weights')
            logger.info('from: {}'.format(cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET))

        elif cfg.WEIGHT_INITIALIZATION.DADA_DEEPLABV2 and not mode == 'val':
            model_params_current = model.state_dict().copy()
            model_params_saved = torch.load(cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET)
            for i in model_params_saved:
                i_parts = i.split(".")
                if not i_parts[1] == "layer5":
                    model_params_current['backbone.{}'.format(".".join(i_parts[1:]))] = model_params_saved[i]
            model.load_state_dict(model_params_current)
            logger.info('ResNet Backbone is initialized with dada deeplab resent pretrained ImageNet weights')
            logger.info('from: {}'.format(cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET))

        elif cfg.WEIGHT_INITIALIZATION.DADA_PRETRAINED and cfg.MODEL_FILE == 'dada_model_new' and not mode == 'val':
            model_params_saved = torch.load(cfg.TRAIN.DADA_PRETRAINED_MODEL_FILE_PATH)
            model_params_current = model.state_dict().copy()
            # init the resent backbone
            for i in model_params_saved['model_state_dict']:
                i_parts = i.split(".")
                if i_parts[0] == "backbone":
                    model_params_current[i] = model_params_saved['model_state_dict'][i]
            logger.info('ResNet Backbone is initialized with dada pretrained weights')
            # init the depth encoder
            if cfg.WEIGHT_INITIALIZATION.DADA_PRETRAINED_DEPTH_ENC:
                model_params_current['aux_enc_depth.enc1.weight'] = model_params_saved['model_state_dict']['decoder.dec1.weight']
                model_params_current['aux_enc_depth.enc1.bias'] = model_params_saved['model_state_dict']['decoder.dec1.bias']
                model_params_current['aux_enc_depth.enc2.weight'] = model_params_saved['model_state_dict']['decoder.dec2.weight']
                model_params_current['aux_enc_depth.enc2.bias'] = model_params_saved['model_state_dict']['decoder.dec2.bias']
                model_params_current['aux_enc_depth.enc3.weight'] = model_params_saved['model_state_dict']['decoder.dec3.weight']
                model_params_current['aux_enc_depth.enc3.bias'] = model_params_saved['model_state_dict']['decoder.dec3.bias']
                logger.info('depth encoder is initialized with dada pretrained weights')
            # init common single conv2d decoder
            if cfg.WEIGHT_INITIALIZATION.DADA_PRETRAINED_SINGLE_CONV_DEC:
                repeat_weight = None
                if cfg.TRAIN.TRAIN_DEPTH_INST_TOGETHER and not cfg.MHA_DADA.ACTIVATE_MHA:
                    if cfg.TRAIN.DEPTH_INST_FEAT_FUSION_TYPE_WHEHN_NO_MHA == 'cat':
                        repeat_weight = True
                elif cfg.TRAIN.TRAIN_DEPTH_INST_TOGETHER and cfg.MHA_DADA.ACTIVATE_MHA:
                    if cfg.MHA_DADA.MODE == 2:
                        repeat_weight = True
                if not repeat_weight:
                    model_params_current['dec_sing_conv.dec.weight'] = model_params_saved['model_state_dict']['decoder.dec4.weight']
                else:
                    model_params_current['dec_sing_conv.dec.weight'] = model_params_saved['model_state_dict']['decoder.dec4.weight'].repeat(1, 2, 1, 1)
                model_params_current['dec_sing_conv.dec.bias'] = model_params_saved['model_state_dict']['decoder.dec4.bias']
                logger.info('single conv2d decoder is initialized with dada pretrained weights')
            # init semanitc head (deeplabv2 classifer)
            if cfg.WEIGHT_INITIALIZATION.DADA_PRETRAINED_SEMANTIC_HEAD:
                for i in model_params_saved['model_state_dict']:
                    i_parts = i.split(".")
                    if i_parts[0] == "semantic_head":
                        model_params_current[i] = model_params_saved['model_state_dict'][i]
                logger.info('semanitc head (deeplabv2 classifer) is initialized with dada pretrained weights')
            model.load_state_dict(model_params_current)
            logger.info('from: {}'.format(cfg.TRAIN.DADA_PRETRAINED_MODEL_FILE_PATH))

            if cfg.FREEZE_BLOCKS.DEPTH_ENC:
                logger.info('Freezing depth encoder ...')
                for param in model.aux_enc_depth.parameters():
                    param.requires_grad = False


        logger.info('cfg.MODEL_SUB_TYPE: {}'.format(cfg.MODEL_SUB_TYPE))
        logger.info('cfg.TRAIN.FREEZE_BN: {}'.format(cfg.TRAIN.FREEZE_BN))
        logger.info('***')
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                if cfg.TRAIN.FREEZE_BN:
                    module.requires_grad_(False)
                else:
                    module.requires_grad_(True)

    elif cfg.MODEL_TYPE == 'dada' and cfg.MODEL_SUB_TYPE == 'deeplabv3':
        raise NotImplementedError('There is no implementation of DADA MTL auxiliary block with DeeplabV3! '
                                  'you can train the deeplabv3 model in DADA style by selecting bsub_scripts/bsub_euler_expid8_0_1_1.sh,'
                                  'but in this case, you have two parallel heads for semantic and depth and'
                                  'there is not MTL auxiliary block for depth.')

    # checkpoint = torch.load(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN)
    # model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = None
    discriminator = None
    optim_state_dict = None
    disc_optim_state_dict = None
    resume_iteration = None
    lr_scheduler_state_dict = None

    # if cfg.MODEL_TYPE == 'panop_v1':
    #     # initialize model
    #     if os.path.isfile(cfg.MODEL.WEIGHTS):
    #         model_weights = torch.load(cfg.MODEL.WEIGHTS)
    #         get_module(model, cfg.DISTRIBUTED).load_state_dict(model_weights, strict=False)
    #         logger.info('Pre-trained model from {}'.format(cfg.MODEL.WEIGHTS))
    #     elif not cfg.MODEL.BACKBONE.PRETRAINED:
    #         if os.path.isfile(cfg.MODEL.BACKBONE.WEIGHTS):
    #             pretrained_weights = torch.load(cfg.MODEL.BACKBONE.WEIGHTS)
    #             get_module(model, cfg.DISTRIBUTED).backbone.load_state_dict(pretrained_weights, strict=False)
    #             logger.info('Pre-trained backbone from {}'.format(cfg.MODEL.BACKBONE.WEIGHTS))
    #         else:
    #             logger.info('No pre-trained weights for backbone, training from scratch.')
    #
    #     # load model
    #     if cfg.TRAIN.RESUME:
    #         model_state_file = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth.tar')
    #         if os.path.isfile(model_state_file):
    #             checkpoint = torch.load(model_state_file)
    #             resume_iteration = checkpoint['iter']
    #             get_module(model, cfg.DISTRIBUTED).load_state_dict(checkpoint['model_state_dict'])
    #             # optimizer.load_state_dict(checkpoint['optimizer'])
    #             optim_state_dict = checkpoint['optimizer_state_dict']
    #             # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #             lr_scheduler_state_dict = checkpoint['lr_scheduler']
    #             logger.info('Loaded checkpoint (starting from iter {})'.format(checkpoint['start_iter']))

    # elif cfg.MODEL_TYPE == 'cvpr2021':
    #     if cfg.WEIGHT_INITIALIZATION.DADA_DEEPLABV2:
    #         model_params_current = model.state_dict().copy()
    #         model_params_saved = torch.load(cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET)
    #         for i in model_params_saved:
    #             i_parts = i.split(".")
    #             if not i_parts[1] == "layer5":
    #                 model_params_current['backbone.{}'.format(".".join(i_parts[1:]))] = model_params_saved[i]
    #         model.load_state_dict(model_params_current)

        # elif cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
        #     if not cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN:
        #         cpfiles = os.listdir(cfg.TRAIN.SNAPSHOT_DIR)
        #         cpfiles = [f for f in cpfiles if '.pth' in f]
        #         cpfiles.sort(reverse=True)
        #         cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, cpfiles[0])
        #     logger.info('Resuming from checkpoint: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
        #     checkpoint = torch.load(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN)
        #     model.load_state_dict(checkpoint['model_state_dict'])
        #     optim_state_dict = checkpoint['optimizer_state_dict']
        #     if not cfg.TRAIN_ISL_FROM_SCRATCH:
        #         resume_iteration = checkpoint['iter']

    disc_inp_dim = None
    disc_inp_dim_2nd = None
    if cfg.ENABLE_DISCRIMINATOR_2ND and cfg.ADV_FEATURE_MODE > 3 and not cfg.APPROACH_TYPE == 'DANDA':
        strk = 'K{}'.format(cfg.ADV_FEATURE_MODE)
        disc_inp_dim = cfg.DISC_INP_DIMS[strk][0]
        disc_inp_dim_2nd = cfg.DISC_INP_DIMS[strk][1]
    elif cfg.ENABLE_DISCRIMINATOR_2ND and cfg.ADV_FEATURE_MODE <= 3 and not cfg.APPROACH_TYPE == 'DANDA':
        raise NotImplementedError('ctrl/utils/common_config_panop.py --> get_model --> when cfg.ENABLE_DISCRIMINATOR_2ND=True then cfg.ADV_FEATURE_MODE should be > 3!!')
    elif cfg.ENABLE_DISCRIMINATOR_2ND and cfg.APPROACH_TYPE == 'DANDA':
        disc_inp_dim = cfg.DISC_INP_DIM
        disc_inp_dim_2nd = disc_inp_dim
    else:
        disc_inp_dim = cfg.DISC_INP_DIM


    if cfg.ENABLE_DISCRIMINATOR:
        if cfg.DISCRIMINATOR_TYPE == 'dada':
            discriminator = get_discriminator(num_classes=disc_inp_dim)
        elif cfg.DISCRIMINATOR_TYPE == 'transformer_encoder':
            encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.TRANSFORMER_ENCODER_D_MODEL, nhead=cfg.TRANSFORMER_ENCODER_NUM_HEAD)
            discriminator = nn.TransformerEncoder(encoder_layer, num_layers=6)

    discriminator2nd = None
    if cfg.ENABLE_DISCRIMINATOR_2ND:
        if cfg.DISCRIMINATOR_TYPE_2ND == 'dada':
            discriminator2nd = get_discriminator(num_classes=disc_inp_dim_2nd)
        elif cfg.DISCRIMINATOR_TYPE_2ND == 'transformer_encoder':
            encoder_layer2nd = nn.TransformerEncoderLayer(d_model=cfg.TRANSFORMER_ENCODER_D_MODEL, nhead=cfg.TRANSFORMER_ENCODER_NUM_HEAD)
            discriminator2nd = nn.TransformerEncoder(encoder_layer2nd, num_layers=6)


        # if cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
        #     discriminator.load_state_dict(checkpoint['disc_state_dict'])
        #     disc_optim_state_dict = checkpoint['disc_optim_state_dict']

    return model, ema_model, discriminator, discriminator2nd, optim_state_dict, disc_optim_state_dict, resume_iteration, lr_scheduler_state_dict


def build_loss_from_cfg(config, logger):

    from ctrl.model_panop.loss import RegularCE, OhemCE, DeepLabCE, L1Loss, MSELoss

    """Builds loss function with specific configuration.
    Args:
        config_panop: the configuration.

    Returns:
        A nn.Module loss.
    """
    if config.NAME == 'cross_entropy':
        # return CrossEntropyLoss(ignore_index=config_panop.IGNORE, reduction='mean')
        return RegularCE(logger, ignore_label=config.IGNORE)
    elif config.NAME == 'ohem':
        return OhemCE(logger, ignore_label=config.IGNORE, threshold=config.THRESHOLD, min_kept=config.MIN_KEPT)
    elif config.NAME == 'hard_pixel_mining':
        return DeepLabCE(logger, ignore_label=config.IGNORE, top_k_percent_pixels=config.TOP_K_PERCENT)
    elif config.NAME == 'dada_sem_loss':
        from ctrl.utils.loss_functions import CrossEntropy2D
        return CrossEntropy2D()
    elif config.NAME == 'mse':
        return MSELoss(reduction=config.REDUCTION)
    elif config.NAME == 'l1':
        return L1Loss(reduction=config.REDUCTION)
    elif config.NAME == 'BerHu':
        from ctrl.utils.loss_functions import BerHuLossDepth
        return BerHuLossDepth()
    elif config.NAME == 'BCE':
        from ctrl.utils.loss_functions import BCELossSS
        return BCELossSS()
    elif config.NAME == 'ce2d_pixelwise_weighted':
        from ctrl.dacs_old.utils.loss import CrossEntropyLoss2dPixelWiseWeighted
        return CrossEntropyLoss2dPixelWiseWeighted(ignore_index=255)
    elif config.NAME == 'hungarian_based_triplet_loss':
        return nn.TripletMarginLoss(margin=0.1, p=2)
    else:
        raise ValueError('Unknown loss type: {}'.format(config.NAME))

def get_criterion_panop(cfg):
    logger = logging.getLogger(__name__)
    criterion_dict = {}
    criterion_dict['semseg'] = build_loss_from_cfg(cfg.LOSS.SEMANTIC, logger)
    if cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
        criterion_dict['center'] = build_loss_from_cfg(cfg.LOSS.CENTER, logger)
        criterion_dict['offset'] = build_loss_from_cfg(cfg.LOSS.OFFSET, logger)
    if cfg.TRAIN.TRAIN_DEPTH_BRANCH:
        criterion_dict['depth'] = build_loss_from_cfg(cfg.LOSS.DEPTH, logger)
    # if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
    #     criterion_dict['disc'] = build_loss_from_cfg(cfg.LOSS.DISC, logger)
    # if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR_2ND:
    #     criterion_dict['disc2nd'] = build_loss_from_cfg(cfg.LOSS.DISC, logger)
    if cfg.TRAIN.TRAIN_WITH_DACS:
        criterion_dict['dacs_unlabeled_loss_semantic'] = build_loss_from_cfg(cfg.LOSS.DACS.UNLABELED_LOSS.SEMANTIC, logger)
    # if cfg.ACTIVATE_DANDA_MEMORY_MODULE:
    #     criterion_dict['hungarian_based_triplet_loss'] = build_loss_from_cfg(cfg.LOSS.HUNGARIAN_BASED_TRIPLET_LOSS, logger)

    return criterion_dict



def get_criterion_panop_v3(cfg):
    logger = logging.getLogger(__name__)
    criterion_dict = {}
    criterion_dict['semseg'] = build_loss_from_cfg(cfg.LOSS.SEMANTIC, logger)
    criterion_dict['center'] = build_loss_from_cfg(cfg.LOSS.CENTER, logger)
    criterion_dict['offset'] = build_loss_from_cfg(cfg.LOSS.OFFSET, logger)
    criterion_dict['depth'] = build_loss_from_cfg(cfg.LOSS.DEPTH, logger)
    criterion_dict['disc'] = build_loss_from_cfg(cfg.LOSS.DISC, logger)
    criterion_dict['disc2nd'] = build_loss_from_cfg(cfg.LOSS.DISC, logger)
    return criterion_dict


def get_criterion_panop_v2(cfg):
    logger = logging.getLogger(__name__)
    criterion_dict = {}
    criterion_dict['semseg'] = build_loss_from_cfg(cfg.LOSS.SEMANTIC, logger)
    criterion_dict['center'] = build_loss_from_cfg(cfg.LOSS.CENTER, logger)
    criterion_dict['offset'] = build_loss_from_cfg(cfg.LOSS.OFFSET, logger)
    criterion_dict['depth'] = build_loss_from_cfg(cfg.LOSS.DEPTH, logger)
    criterion_dict['dacs_unlabeled_loss_semantic'] = build_loss_from_cfg(cfg.LOSS.DACS.UNLABELED_LOSS.SEMANTIC, logger)
    if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
        criterion_dict['disc'] = build_loss_from_cfg(cfg.LOSS.DISC, logger)
    if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR_2ND:
        criterion_dict['disc2nd'] = build_loss_from_cfg(cfg.LOSS.DISC, logger)
    if cfg.ACTIVATE_DANDA_MEMORY_MODULE:
        criterion_dict['hungarian_based_triplet_loss'] = build_loss_from_cfg(cfg.LOSS.HUNGARIAN_BASED_TRIPLET_LOSS, logger)

    return criterion_dict

def get_criterion():
    logger = logging.getLogger(__name__)
    criterion_dict = {}
    from ctrl.utils.loss_functions import CrossEntropy2D
    criterion_dict['semseg'] = CrossEntropy2D()
    from ctrl.utils.loss_functions import BerHuLossDepth
    criterion_dict['depth'] = BerHuLossDepth()
    from ctrl.utils.loss_functions import BCELossSS
    criterion_dict['disc_loss'] = BCELossSS()
    return criterion_dict

def get_optimizer_dacs_panop_padnet(cfg, model, USeDataParallel=None, discriminator=None, discriminator2nd=None, optim_state_dict=None, disc_optim_state_dict=None, train_only_dacs_ori=False):
    logger = logging.getLogger(__name__)
    logger.info('get_optimizer_dacs_panop_padnet() --> ctrl/utils/common_config_panop.py')

    modeL = get_module(model, cfg.DISTRIBUTED)
    # Backbone
    optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)

    if cfg.INCLUDE_DADA_AUXBLOCK:
        optim_list_dada_aux_encoder = modeL.dada_aux_encoder.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_dada_aux_decoder = modeL.dada_aux_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
    else:
        optim_list_dada_aux_encoder = []
        optim_list_dada_aux_decoder = []

    # padnet blocks
    optim_list_init_task = modeL.initial_task_prediction_heads.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_attention = modeL.multi_modal_distillation.optim_parameters(cfg.SOLVER.BASE_LR)

    from ctrl.model.padnet.layers import get_task_dict
    tn_dict = get_task_dict()
    task_names = []
    for tn in cfg.TASKNAMES:
        task_names.append(tn_dict[tn])
    optim_list_task_heads = []
    for task in task_names + ["D_src"]:
        optim_list_task_heads.append(modeL.heads[task].optim_parameters(cfg.SOLVER.BASE_LR))

    if "I" in task_names:
        optim_list_center_sub_head = modeL.center_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_offset_sub_head = modeL.offset_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
    else:
        optim_list_center_sub_head = []
        optim_list_offset_sub_head = []

    # optim list
    optim_list = optim_list_backbone + \
                 optim_list_dada_aux_encoder + optim_list_dada_aux_decoder + \
                 optim_list_init_task + optim_list_attention + \
                 optim_list_center_sub_head + optim_list_offset_sub_head
    # optim_list_task_heads[0] + optim_list_task_heads[1] + optim_list_task_heads[2] + optim_list_task_heads[3] + \

    for i in range(len(optim_list_task_heads)):
        optim_list += optim_list_task_heads[i]

    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer


def get_optimizer_dacs_panop_v3(cfg, model, USeDataParallel=None, discriminator=None, discriminator2nd=None, optim_state_dict=None, disc_optim_state_dict=None, train_only_dacs_ori=False):
    logger = logging.getLogger(__name__)
    logger.info('get_optimizer_dacs_panop_v3() --> ctrl/utils/common_config_panop.py')

    modeL = get_module(model, cfg.DISTRIBUTED)

    # Backbone
    optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_dada_aux_encoder = modeL.dada_aux_encoder.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_dada_aux_decoder = modeL.dada_aux_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
    # semantic head
    optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # instance head
    optim_list_instance_head = modeL.instance_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_center_sub_head = modeL.center_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_offset_sub_head = modeL.offset_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # depth head
    optim_list_depth_head = modeL.depth_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_depth_head_src = modeL.depth_head_src.optim_parameters(cfg.SOLVER.BASE_LR)

    # optim list
    optim_list = optim_list_backbone + optim_list_semantic_head + optim_list_instance_head + \
                 optim_list_center_sub_head + optim_list_offset_sub_head + \
                 optim_list_dada_aux_encoder + optim_list_dada_aux_decoder + optim_list_depth_head + optim_list_depth_head_src

    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer


def get_optimizer_dacs_panop_v2_ablation(cfg, model, USeDataParallel=None, discriminator=None, discriminator2nd=None, optim_state_dict=None, disc_optim_state_dict=None, train_only_dacs_ori=False):
    logger = logging.getLogger(__name__)
    logger.info('get_optimizer_dacs_panop_v2_ablation() --> ctrl/utils/common_config_panop.py')

    modeL = get_module(model, cfg.DISTRIBUTED)
    # Backbone
    optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)

    optim_list_dada_aux_encoder = []
    optim_list_dada_aux_decoder = []
    if cfg.INCLUDE_DADA_AUXBLOCK:
        optim_list_dada_aux_encoder = modeL.dada_aux_encoder.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_dada_aux_decoder = modeL.dada_aux_decoder.optim_parameters(cfg.SOLVER.BASE_LR)

    # semantic head
    optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)

    # instance head
    optim_list_instance_head = modeL.instance_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_center_sub_head = modeL.center_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_offset_sub_head = modeL.offset_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)

    optim_list_depth_head = []
    # depth head
    if cfg.TRAIN.TRAIN_DEPTH_BRANCH and not cfg.DADA_STYLE_DEPTH_HEAD:
        optim_list_depth_head = modeL.depth_head.optim_parameters(cfg.SOLVER.BASE_LR)

    # optim list
    optim_list = optim_list_backbone + optim_list_semantic_head + optim_list_instance_head + \
                 optim_list_center_sub_head + optim_list_offset_sub_head + \
                 optim_list_dada_aux_encoder + optim_list_dada_aux_decoder + optim_list_depth_head

    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer


def get_optimizer_panop_dada(cfg, model, USeDataParallel=None, discriminator=None, discriminator2nd=None, optim_state_dict=None, disc_optim_state_dict=None, train_only_dacs_ori=False):
    logger = logging.getLogger(__name__)
    logger.info('get_optimizer_panop_dada() --> ctrl/utils/common_config_panop.py')

    modeL = get_module(model, cfg.DISTRIBUTED)
    # Backbone
    optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
    # DADA  aux encoder-decoder block
    optim_list_dada_aux_encoder = modeL.dada_aux_encoder.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_dada_aux_decoder = modeL.dada_aux_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
    # semantic head
    optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # instance head
    optim_list_instance_head = modeL.instance_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_center_sub_head = modeL.center_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_offset_sub_head = modeL.offset_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # depth head
    # optim_list_depth_head = modeL.depth_head.optim_parameters(cfg.SOLVER.BASE_LR)

    # optim list
    optim_list = optim_list_backbone + optim_list_semantic_head + optim_list_instance_head + \
                 optim_list_center_sub_head + optim_list_offset_sub_head + \
                 optim_list_dada_aux_encoder + optim_list_dada_aux_decoder

    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer

def get_optimizer_dacs_panop_v2(cfg, model, USeDataParallel=None, discriminator=None, discriminator2nd=None, optim_state_dict=None, disc_optim_state_dict=None, train_only_dacs_ori=False):
    logger = logging.getLogger(__name__)
    logger.info('get_optimizer_dacs_panop_v2() --> ctrl/utils/common_config_panop.py')

    modeL = get_module(model, cfg.DISTRIBUTED)
    optim_list_dada_aux_encoder = []
    optim_list_dada_aux_decoder = []
    # Backbone
    optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_dada_aux_encoder = modeL.dada_aux_encoder.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_dada_aux_decoder = modeL.dada_aux_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
    # semantic head
    optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # instance head
    optim_list_instance_head = modeL.instance_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_center_sub_head = modeL.center_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_offset_sub_head = modeL.offset_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # depth head
    optim_list_depth_head = modeL.depth_head.optim_parameters(cfg.SOLVER.BASE_LR)

    # optim list
    optim_list = optim_list_backbone + optim_list_semantic_head + optim_list_instance_head + \
                 optim_list_center_sub_head + optim_list_offset_sub_head + \
                 optim_list_dada_aux_encoder + optim_list_dada_aux_decoder + optim_list_depth_head

    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer

def get_optimizer_dacs_panop(cfg, model, USeDataParallel=None, discriminator=None, discriminator2nd=None, optim_state_dict=None, disc_optim_state_dict=None, train_only_dacs_ori=False):
    logger = logging.getLogger(__name__)
    logger.info('get_optimizer_dacs_panop() --> ctrl/utils/common_config_panop.py')

    modeL = get_module(model, cfg.DISTRIBUTED)
    optim_list_dada_aux_encoder = []
    optim_list_dada_aux_decoder = []
    # Backbone
    optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
    if cfg.INCLUDE_DADA_AUXBLOCK:
        optim_list_dada_aux_encoder = modeL.dada_aux_encoder.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_dada_aux_decoder = modeL.dada_aux_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
    # semantic head
    optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # instance head
    optim_list_instance_head = modeL.instance_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_center_sub_head = modeL.center_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_offset_sub_head = modeL.offset_sub_head.optim_parameters(cfg.SOLVER.BASE_LR)

    # optim list
    optim_list = optim_list_backbone + optim_list_semantic_head + optim_list_instance_head + optim_list_center_sub_head + optim_list_offset_sub_head + optim_list_dada_aux_encoder + optim_list_dada_aux_decoder

    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer



def get_optimizer_dacs(cfg, model, USeDataParallel=None, discriminator=None, discriminator2nd=None, optim_state_dict=None, disc_optim_state_dict=None, train_only_dacs_ori=False):
    logger = logging.getLogger(__name__)
    logger.info('get_optimizer_dada() --> ctrl/utils/common_config_panop.py')

    modeL = get_module(model, cfg.DISTRIBUTED)
    # Backbone
    optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
    # semantic head
    optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # optim list
    optim_list = optim_list_backbone + \
                 optim_list_semantic_head
    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer


def get_optimizer_dada(cfg, model, USeDataParallel=None, discriminator=None, discriminator2nd=None, optim_state_dict=None, disc_optim_state_dict=None):
    logger = logging.getLogger(__name__)
    logger.info('get_optimizer_dada() --> ctrl/utils/common_config_panop.py')
    optim_list_aux_enc_depth = []
    optim_list_aux_enc_inst = []
    optim_list_center_head = []
    optim_list_offset_head = []
    optim_list_mha = []
    optim_list_memory_module = []
    optimizer_discriminator = None
    optimizer_discriminator2nd = None
    modeL = get_module(model, cfg.DISTRIBUTED)
    # Backbone
    optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
    # Depth encoder
    if cfg.TRAIN.TRAIN_DEPTH_BRANCH:
        optim_list_aux_enc_depth = modeL.aux_enc_depth.optim_parameters(cfg.SOLVER.BASE_LR)
    # Instance encoder
    if cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
        optim_list_aux_enc_inst = modeL.aux_enc_inst.optim_parameters(cfg.SOLVER.BASE_LR)
        if not cfg.TRAIN.CENTER_HEAD_DADA_STYLE:
            optim_list_center_head = modeL.center_head.optim_parameters(cfg.SOLVER.BASE_LR)
        if cfg.TRAIN.TRAIN_OFFSET_HEAD:
            optim_list_offset_head = modeL.offset_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # Single Conv decoder
    optim_list_dec_sing_conv = modeL.dec_sing_conv.optim_parameters(cfg.SOLVER.BASE_LR)
    # semantic head
    optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # MHA
    if cfg.MHA_DADA.ACTIVATE_MHA:
        optim_list_mha = modeL.mha.optim_parameters(cfg.SOLVER.BASE_LR)

    if cfg.ACTIVATE_DANDA_MEMORY_MODULE:
        optim_list_memory_module = modeL.get_memory_block_paprams(cfg.SOLVER.BASE_LR)

    # optim list
    optim_list = optim_list_backbone + \
                 optim_list_aux_enc_depth + \
                 optim_list_aux_enc_inst + \
                 optim_list_center_head + \
                 optim_list_offset_head + \
                 optim_list_dec_sing_conv + \
                 optim_list_semantic_head + \
                 optim_list_mha + \
                 optim_list_memory_module
    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if cfg.ENABLE_DISCRIMINATOR:
        print('')
        print('*** ctrl/utils/common_config_panop.py --> get_optimizer_dada() ')
        print('cfg.SOLVER.ADAM_BETAS_DISC_1ST: {}'.format(cfg.SOLVER.ADAM_BETAS_DISC_1ST))
        print('')
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=cfg.SOLVER.DISC_LR, betas=cfg.SOLVER.ADAM_BETAS_DISC_1ST)

    if cfg.ENABLE_DISCRIMINATOR_2ND:
        optimizer_discriminator2nd = torch.optim.Adam(discriminator2nd.parameters(), lr=cfg.SOLVER.DISC_LR_2ND, betas=cfg.SOLVER.ADAM_BETAS_DISC_2ND)

    return optimizer, optimizer_discriminator, optimizer_discriminator2nd


def get_optimizer_dada_old(cfg, model, USeDataParallel=None, discriminator=None, optim_state_dict=None, disc_optim_state_dict=None):
    logger = logging.getLogger(__name__)
    logger.info('get_optimizer_dada() --> ctrl/utils/common_config_panop.py')
    modeL = get_module(model, cfg.DISTRIBUTED)
    optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_decoder = modeL.decoder.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)
    # optim_list_depth_head = modeL.depth_head.optim_parameters(cfg.SOLVER.BASE_LR) # dada does not have depth paparameters
    if cfg.MHA.ACTIVATE_MHA and cfg.MHA.TYPE == 'BACKBONE_FEAT':
        optim_list_mha_within_feat = modeL.mha_within_feat.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list = optim_list_backbone + optim_list_decoder + optim_list_semantic_head + optim_list_mha_within_feat
    else:
        optim_list = optim_list_backbone + optim_list_decoder + optim_list_semantic_head
    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if cfg.ENABLE_DISCRIMINATOR:
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=cfg.SOLVER.DISC_LR, betas=cfg.SOLVER.ADAM_BETAS_DISC_1ST)
    if not cfg.TRAIN_ISL_FROM_SCRATCH:
        if optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
            optimizer.load_state_dict(optim_state_dict)
            logger.info('model optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
        if cfg.ENABLE_DISCRIMINATOR:
            if disc_optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
                optimizer_discriminator.load_state_dict(disc_optim_state_dict)
                logger.info('discriminator optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
    return optimizer, optimizer_discriminator




def get_optimizer_panop_v1(cfg, model, USeDataParallel=None, discriminator=None, optim_state_dict=None, disc_optim_state_dict=None):
    logger = logging.getLogger(__name__)
    optimizer = None
    optimizer_discriminator = None

    if True: # elif cfg.DADA_STYLE_LR and cfg.MODEL_SUB_TYPE == 'deeplabv3':
        logger.info('setting the lr for model parameters as per DADA style')
        optim_list_depth_decoder = []
        optim_list_depth_head = []
        optim_list_dada_aux_encoder = []
        optim_list_dada_aux_decoder = []
        modeL = get_module(model, cfg.DISTRIBUTED)
        optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_semantic_decoder = modeL.semantic_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)
        if True:  # cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
            optim_list_instance_decoder = modeL.instance_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
            optim_list_instance_head = modeL.instance_head.optim_parameters(cfg.SOLVER.BASE_LR)
        if cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            optim_list_depth_decoder = modeL.depth_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
            optim_list_depth_head = modeL.depth_head.optim_parameters(cfg.SOLVER.BASE_LR)
        if cfg.INCLUDE_DADA_AUXBLOCK:
            optim_list_dada_aux_encoder = modeL.dada_aux_encoder.optim_parameters(cfg.SOLVER.BASE_LR)
            optim_list_dada_aux_decoder = modeL.dada_aux_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list = optim_list_backbone + optim_list_semantic_decoder + optim_list_semantic_head + \
                     optim_list_instance_decoder + optim_list_instance_head + \
                     optim_list_depth_decoder + optim_list_depth_head  + \
                     optim_list_dada_aux_encoder + optim_list_dada_aux_decoder
        optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        return optimizer, optimizer_discriminator

    # if cfg.PANOPTIC_DEEPLAB_STYLE_LR:
    #     logger.info('Setting the lr for model parameters as per panoptic-deeplab style')
    #     from ctrl.solver_panop import build_optimizer
    #     optimizer = build_optimizer(cfg, model)
    #     if cfg.ENABLE_DISCRIMINATOR:
    #         optimizer_discriminator = build_optimizer(cfg, discriminator)
    # elif cfg.DADA_STYLE_LR and cfg.MODEL_SUB_TYPE == 'deeplabv2':
    #     logger.info('setting the lr for model parameters as per DADA style')
    #     optim_list_instance_deeplabv2_decoder = []
    #     optim_list_instance_head = []
    #     optim_list_depth_deeplabv2_decoder = []
    #     optim_list_depth_head = []
    #     modeL = get_module(model, cfg.DISTRIBUTED)
    #     optim_list_backbone = modeL.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
    #     optim_list_auxBlock = modeL.aux_block.optim_parameters(cfg.SOLVER.BASE_LR)
    #     optim_list_semantic_deeplabv2_decoder = modeL.semantic_deeplabv2_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
    #     optim_list_semantic_head = modeL.semantic_head.optim_parameters(cfg.SOLVER.BASE_LR)
    #     if cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
    #         optim_list_instance_deeplabv2_decoder = modeL.instance_deeplabv2_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
    #         optim_list_instance_head = modeL.instance_head.optim_parameters(cfg.SOLVER.BASE_LR)
    #     if cfg.TRAIN.TRAIN_DEPTH_BRANCH:
    #         optim_list_depth_deeplabv2_decoder = modeL.depth_deeplabv2_decoder.optim_parameters(cfg.SOLVER.BASE_LR)
    #         optim_list_depth_head = modeL.depth_head.optim_parameters(cfg.SOLVER.BASE_LR)
    #
    #     optim_list = optim_list_backbone + optim_list_auxBlock + \
    #                  optim_list_semantic_deeplabv2_decoder + optim_list_semantic_head + \
    #                  optim_list_instance_deeplabv2_decoder + optim_list_instance_head + \
    #                  optim_list_depth_deeplabv2_decoder + optim_list_depth_head
    #
    #     optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    #     if cfg.ENABLE_DISCRIMINATOR:
    #         optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=cfg.SOLVER.DISC_LR, betas=(0.9, 0.99))
    #     if not cfg.TRAIN_ISL_FROM_SCRATCH:
    #         if optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
    #             optimizer.load_state_dict(optim_state_dict)
    #             logger.info('model optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
    #         if cfg.ENABLE_DISCRIMINATOR:
    #             if disc_optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
    #                 optimizer_discriminator.load_state_dict(disc_optim_state_dict)
    #                 logger.info('discriminator optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))





def get_optimizer(cfg, model, logger, USeDataParallel=None, discriminator=None, optim_state_dict=None, disc_optim_state_dict=None):
    optimizer_discriminator = None
    if USeDataParallel:
        optim_list_backbone = model.module.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_encoder = model.module.encoder.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_decoder_single_conv = model.module.decoder_single_conv.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_decoder_semseg = model.module.decoder_semseg.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_depth_head = model.module.depth_head.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_decoder_semseg_given_depth = model.module.decoder_semseg_given_depth.optim_parameters(cfg.SOLVER.BASE_LR)
    else:
        optim_list_backbone = model.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_encoder = model.encoder.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_decoder_single_conv = model.decoder_single_conv.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_decoder_semseg = model.decoder_semseg.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_depth_head = model.module.depth_head.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_decoder_semseg_given_depth = model.decoder_semseg_given_depth.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list = optim_list_backbone + optim_list_encoder + optim_list_decoder_single_conv + \
                 optim_list_decoder_semseg + optim_list_depth_head + optim_list_decoder_semseg_given_depth
    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.ENABLE_DISCRIMINATOR:
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=cfg.SOLVER.DISC_LR, betas=(0.9, 0.99))
    if not cfg.TRAIN_ISL_FROM_SCRATCH:
        if optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
            optimizer.load_state_dict(optim_state_dict)
            logger.info('model optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
        if cfg.ENABLE_DISCRIMINATOR:
            if disc_optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
                optimizer_discriminator.load_state_dict(disc_optim_state_dict)
                logger.info('discriminator optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
    return optimizer, optimizer_discriminator



def get_optimizer_v2(cfg, model, USeDataParallel=None, discriminator=None, optim_state_dict=None, disc_optim_state_dict=None):

    logger = logging.getLogger(__name__)

    optimizer_discriminator = None
    if USeDataParallel:
        optim_list_backbone = model.module.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_decoder = model.module.decoder.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_head_seg = model.module.head_seg.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_head_dep = model.module.head_dep.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_head_srh = model.module.head_srh.optim_parameters(cfg.SOLVER.BASE_LR)
    else:
        optim_list_backbone = model.backbone.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_decoder = model.decoder.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_head_seg = model.head_seg.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_head_dep = model.head_dep.optim_parameters(cfg.SOLVER.BASE_LR)
        optim_list_head_srh = model.head_srh.optim_parameters(cfg.SOLVER.BASE_LR)
    optim_list = optim_list_backbone + optim_list_decoder + optim_list_head_seg + optim_list_head_dep + optim_list_head_srh
    optimizer = torch.optim.SGD(optim_list, lr=cfg.SOLVER.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.ENABLE_DISCRIMINATOR:
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=cfg.SOLVER.DISC_LR, betas=(0.9, 0.99))
    if not cfg.TRAIN_ISL_FROM_SCRATCH:
        if optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
            optimizer.load_state_dict(optim_state_dict)
            logger.info('model optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
        if cfg.ENABLE_DISCRIMINATOR:
            if disc_optim_state_dict and cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT:
                optimizer_discriminator.load_state_dict(disc_optim_state_dict)
                logger.info('discriminator optimizer is loaded from: {}'.format(cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN))
    return optimizer, optimizer_discriminator




def worker_init_reset_seed(worker_id):
    logger = logging.getLogger(__name__)
    logger.info('*** ctrl/utils/common_config_panop.py --> worker_init_reset_seed(worker_id) ***')
    randId = np.random.randint(2 ** 31)
    seed = randId + worker_id
    logger.info('[randId: {} + worker_id: {}] = seed: {}'.format(randId, worker_id, seed))
    seed_all_rng(seed)


# a simple custom collate function, just to show the idea
def my_collate(batch):
    '''
    batch is a list of 2 element
    batch=[0]  a tuple of 4 elements
    batch[0][0] = ndarray N, 3, 512, 512 (image)
    batch[0][1] =  dict 10 elements - this is label_panop_dict
    batch[0][2] = ndarray - shape of one image in the batch e.g. 760, 1280, 3
    batch[0][3] = string image name

    images_source.shape = torch.Size([1, 3, 512, 512])
    images_source.type() = 'torch.cuda.FloatTensor'

    label_panop_dict['semanitc'].shape = {Tensor:(1, 512, 512)}
    label_panop_dict['semantic'].type() = 'torch.cuda.LongTensor'

    label_panop_dict['center'].shape = torch.Size([1, 1, 512, 512])
    label_panop_dict['center'].type() = 'torch.cuda.FloatTensor'

    label_panop_dict['offset'].shape = torch.Size([1, 2, 512, 512])
    label_panop_dict['offset'].type() = 'torch.cuda.FloatTensor'

    label_panop_dict['depth'].shape = torch.Size([1, 512, 512])
    label_panop_dict['depth'].type() = 'torch.cuda.FloatTensor'
    '''
    image = [item[0] for item in batch]
    image = np.stack(image, axis=0)
    image = torch.FloatTensor(torch.from_numpy(image))
    sl = [] # semantic label
    cl = [] # center label
    ol = [] # offset label
    dl = [] # depth label
    for b in batch:
        lpd = b[1] # label_panop_dict
        sl.append(lpd['semantic'])
        cl.append(lpd['center'])
        ol.append(lpd['offset'])
        dl.append(lpd['depth'])
    sl = np.stack(sl, axis=0)
    sl = torch.LongTensor(torch.from_numpy(sl))
    cl = np.stack(cl, axis=0)
    cl = torch.FloatTensor(torch.from_numpy(cl))
    ol = np.stack(ol, axis=0)
    ol = torch.FloatTensor(torch.from_numpy(ol))
    dl = np.stack(dl, axis=0)
    dl = torch.FloatTensor(torch.from_numpy(dl))
    label_panop_dict = {}
    label_panop_dict['semantic'] = sl
    label_panop_dict['center'] = cl
    label_panop_dict['offset'] = ol
    label_panop_dict['depth'] = dl
    shape = [item[2] for item in batch]
    name = [item[3] for item in batch]
    return [image, label_panop_dict, shape, name]


def get_data_loaders(cfg, get_target_train_loader=True):
    logger = logging.getLogger(__name__)
    from ctrl.dataset import panoptic_deeplab_sampler as samplers
    from ctrl.utils.panoptic_deeplab.comm import get_world_size
    num_workers = get_world_size()
    images_per_batch = cfg.TRAIN.IMS_PER_BATCH
    assert (images_per_batch % num_workers == 0), "TRAIN.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".\
                                                    format(images_per_batch, num_workers)
    assert (images_per_batch >= num_workers), "TRAIN.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".\
                                                format(images_per_batch, num_workers)
    images_per_worker = images_per_batch // num_workers   # e.g. 2 = 8 / 4
    logger.info('num_workers: {}; images_per_batch: {}; images_per_worker: {}'.format(num_workers, images_per_batch, images_per_worker))
    source_train_dataset = None
    target_train_dataset = None
    target_test_dataset = None
    source_train_loader = None
    target_train_loader = None
    source_train_nsamp = None
    target_train_nsamp = None
    target_test_nsamp = None
    target_val_loader = None

    batch_sampler_source = None

    IMG_MEAN = np.array(cfg.TRAIN.IMG_MEAN, dtype=np.float32)

    if cfg.SOURCE == 'SYNTHIA':

        from ctrl.dataset.synthia_panop import SYNTHIADataSetDepth

        source_train_dataset = SYNTHIADataSetDepth(
            root=cfg.DATA_DIRECTORY_SOURCE,
            list_path=cfg.DATA_LIST_SOURCE,
            set=cfg.TRAIN.SET_SOURCE,
            num_classes=cfg.NUM_CLASSES,
            max_iters=None,
            crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
            mean=IMG_MEAN,
            use_depth=cfg.USE_DEPTH,
            depth_processing=cfg.DEPTH_PROCESSING,
            cfg=cfg,
            joint_transform=None,
        )
        sampler_name_source = cfg.DATALOADER.SAMPLER_TRAIN_SOURCE
        logger.info("Using training sampler {}".format(sampler_name_source))
        if sampler_name_source == "TrainingSamplerSource":
            sampler_source = samplers.TrainingSampler(len(source_train_dataset), shuffle=cfg.DATALOADER.TRAIN_SHUFFLE)
        else:
            raise ValueError("Unknown training sampler_source: {}".format(sampler_name_source))
        batch_sampler_source = torch.utils.data.sampler.BatchSampler(sampler_source, images_per_worker, drop_last=True)
    if cfg.SOURCE:
        if cfg.MODEL_TYPE == 'dacs_old' and cfg.TRAIN.IMS_PER_BATCH > 1:
            source_train_loader = data.DataLoader(
                source_train_dataset,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                batch_sampler=batch_sampler_source,
                worker_init_fn=worker_init_reset_seed,
                collate_fn=my_collate,
                pin_memory=True,
            )
        else:
            source_train_loader = data.DataLoader(
                source_train_dataset,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                batch_sampler=batch_sampler_source,
                worker_init_fn=worker_init_reset_seed,
                pin_memory=True,
            )

    if cfg.TARGET == 'Cityscapes':
        from ctrl.dataset.cityscapes_panop_sep25 import CityscapesDataSet
        # from ctrl.dataset.cityscapes_panop import CityscapesDataSet

        if get_target_train_loader:
            target_train_dataset = CityscapesDataSet(
                root=cfg.DATA_DIRECTORY_TARGET,
                list_path=cfg.DATA_LIST_TARGET,
                set=cfg.TRAIN.SET_TARGET,
                info_path=cfg.TRAIN.INFO_TARGET,
                max_iters=None,
                crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                mean=IMG_MEAN,
                joint_transform=None,
                cfg=cfg,
            )
        target_test_dataset = CityscapesDataSet(
            root=cfg.DATA_DIRECTORY_TARGET,
            list_path=cfg.DATA_LIST_TARGET,
            set=cfg.TEST.SET_TARGET,
            info_path=cfg.TEST.INFO_TARGET,
            crop_size=cfg.TEST.INPUT_SIZE_TARGET,
            mean=cfg.TEST.IMG_MEAN,
            labels_size=cfg.TEST.OUTPUT_SIZE_TARGET,
            joint_transform=None,
            cfg=cfg,
        )

    if get_target_train_loader:
        sampler_name_target_train = cfg.DATALOADER.SAMPLER_TRAIN_TARGET
        logger.info("Using training sampler {}".format(sampler_name_target_train))
        if sampler_name_target_train == "TrainingSamplerTarget":
            sampler_target_train = samplers.TrainingSampler(len(target_train_dataset), shuffle=cfg.DATALOADER.TRAIN_SHUFFLE)
        else:
            raise ValueError("Unknown training sampler_target: {}".format(sampler_name_target_train))
        batch_sampler_target_train = torch.utils.data.sampler.BatchSampler(sampler_target_train, images_per_worker, drop_last=True)
        if cfg.MODEL_TYPE == 'dacs_old' and cfg.TRAIN.IMS_PER_BATCH > 1:
            target_train_loader = data.DataLoader(
                target_train_dataset,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                batch_sampler=batch_sampler_target_train,
                worker_init_fn=worker_init_reset_seed,
                collate_fn=my_collate,
                pin_memory=True,
            )
        else:
            target_train_loader = data.DataLoader(
                target_train_dataset,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                batch_sampler=batch_sampler_target_train,
                worker_init_fn=worker_init_reset_seed,
                pin_memory=True,
            )

    sampler_name_target_test = cfg.DATALOADER.SAMPLER_TEST_TARGET
    logger.info("Using test sampler {}".format(sampler_name_target_test))
    if sampler_name_target_test == "TestingSamplerTarget":
        sampler_target_test = samplers.InferenceSampler(len(target_test_dataset))
    else:
        raise ValueError("Unknown testing sampler_target: {}".format(sampler_name_target_test))

    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in panoptic deeplab papers.
    batch_sampler_target_test = torch.utils.data.sampler.BatchSampler(sampler_target_test, 1, drop_last=False)
    target_val_loader = data.DataLoader(
        target_test_dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler_target_test,
        pin_memory=True,
    )
    if cfg.SOURCE:
        source_train_nsamp = len(source_train_dataset)
        logger.info('{} : source train examples: {}'.format(cfg.SOURCE, source_train_nsamp))
    if target_train_dataset:
        target_train_nsamp = len(target_train_dataset)
    target_test_nsamp = len(target_test_dataset)
    if target_train_dataset:
        logger.info('{} : target train examples: {}'.format(cfg.TARGET, target_train_nsamp))
    logger.info('{} : target test examples: {}'.format(cfg.TARGET, target_test_nsamp))

    return source_train_loader, target_train_loader, target_val_loader, source_train_nsamp, target_train_nsamp, target_test_nsamp



# from .meta_arch import PanopticDeepLab
# from ctrl.model_panop.meta_arch import PanopticDeepLab
# from ctrl.model_panop.backbone import resnet101



# PANOPITC DEEPLAB CVPR 2020 MODEL DEFINITATION
    # backbone = resnet101(pretrained=True, replace_stride_with_dilation=(False, False, False))
    # panoptic_deeplab = dict(
    #     replace_stride_with_dilation=cfg.MODEL.BACKBONE.DILATION,
    #     in_channels=cfg.MODEL.DECODER.IN_CHANNELS,
    #     feature_key=cfg.MODEL.DECODER.FEATURE_KEY,
    #     low_level_channels=cfg.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_CHANNELS,
    #     low_level_key=cfg.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_KEY,
    #     low_level_channels_project=cfg.MODEL.PANOPTIC_DEEPLAB.LOW_LEVEL_CHANNELS_PROJECT,
    #     decoder_channels=cfg.MODEL.DECODER.DECODER_CHANNELS,
    #     atrous_rates=cfg.MODEL.DECODER.ATROUS_RATES,
    #     num_classes=cfg.DATASET.NUM_CLASSES,
    #     has_instance=cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.ENABLE,
    #     instance_low_level_channels_project=cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.LOW_LEVEL_CHANNELS_PROJECT,
    #     instance_decoder_channels=cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.DECODER_CHANNELS,
    #     instance_head_channels=cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.HEAD_CHANNELS,
    #     instance_aspp_channels=cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.ASPP_CHANNELS,
    #     instance_num_classes=cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.NUM_CLASSES,
    #     instance_class_key=cfg.MODEL.PANOPTIC_DEEPLAB.INSTANCE.CLASS_KEY,
    #     semantic_loss=build_loss_from_cfg(cfg.LOSS.SEMANTIC),
    #     semantic_loss_weight = cfg.LOSS.SEMANTIC.WEIGHT,
    #     center_loss = build_loss_from_cfg(cfg.LOSS.CENTER),
    #     center_loss_weight = cfg.LOSS.CENTER.WEIGHT,
    #     offset_loss = build_loss_from_cfg(cfg.LOSS.OFFSET),
    #     offset_loss_weight = cfg.LOSS.OFFSET.WEIGHT,
    # )
    # model = PanopticDeepLab(backbone, **panoptic_deeplab)
    # set batchnorm momentum


# def setup_exp_params(cfg, cmdline_inputs):
#
#     if cfg.DEBUG:   # TODO:DONE
#         cfg.MACHINE = 0
#         cfg.GPUS = [0]
#         cfg.DATALOADER.NUM_WORKERS = 0
#         cfg.TRAIN.IMS_PER_BATCH = 1
#
#     # if training on multiple gpus, then it uses DistributedDataParallel
#     gpus = list(cfg.GPUS) # TODO:DONE
#
#     cfg.DISTRIBUTED = len(gpus) > 1 # TODO:DONE
#     cfg.DEVICE = torch.device('cuda:{}'.format(cmdline_inputs.local_rank)) # TODO:DONE
#
#     # TODO:DONE
#     cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
#     cfg.DISC_INP_DIM = (cfg.NUM_CLASSES * 2) + cfg.NUM_DEPTH_BINS  # 16+16+15 = 47 ; or 7+7+15 = 29
#     cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
#     cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
#
#     # TODO:DONE
#     if cfg.RESO == 'LOW':
#        cfg.TRAIN.INPUT_SIZE_SOURCE = (640, 320)
#        cfg.TRAIN.INPUT_SIZE_TARGET = (640, 320)
#        cfg.TEST.INPUT_SIZE_TARGET = (640, 320)
#        cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
#     elif cfg.RESO == 'FULL':
#        cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 760)
#        cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)
#        cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
#        cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
#     if cfg.DEBUG:
#         # cfg.NUM_WORKERS = 0
#         cfg.NUM_WORKERS_TEST = 0
#         cfg.TRAIN.DISPLAY_LOSS_RATE = 1
#         cfg.TRAIN.EVAL_EVERY = 99999999
#         cfg.TRAIN.SAVE_PRED_EVERY = 10
#         cfg.TRAIN.TENSORBOARD_VIZRATE = 10
#         if not cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL and not cfg.PANOPTIC_DEEPLAB_DATA_AUG_ONLY_NORM:
#             cfg.TRAIN.INPUT_SIZE_SOURCE = (640, 320)
#             cfg.TRAIN.INPUT_SIZE_TARGET = (640, 320)
#             cfg.TEST.INPUT_SIZE_TARGET = (640, 320)
#             cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
#         cfg.TEST.NUM_TEST_SAMPLES = 1
#
#     if cfg.DISTRIBUTED:
#         cfg.DATALOADER.NUM_WORKERS = 4
#     else:
#         cfg.DATALOADER.NUM_WORKERS = 0
#
#     if cfg.TARGET == 'Mapillary':
#         cfg.TEST.OUTPUT_SIZE_TARGET = None
#     if cfg.TARGET == 'Mapillary' and cfg.RESO == 'FULL':
#         cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 768)
#         cfg.TEST.INPUT_SIZE_TARGET = (1024, 768)
#     # experiment related params
#     if cfg.TARGET == 'Mapillary':
#         cfg.EXP_SETUP = 'SYNTHIA_TO_MAPILLARY'
#     else:
#         cfg.EXP_SETUP = 'SYNTHIA_TO_CITYSCAPES'
#
#
#     # ------------------------------------------------------------------------
#     # create yml file for oracle training without data augmentation
#     # when you use oracel training set cfg.TRAIN_ONLY_SOURCE= True
#     # ------------------------------------------------------------------------
#     # training the Oracle, e.g., train and test on cityscape
#     # then consider the target as source and there is no target anyway
#     if cfg.TRAIN_ORACLE and cfg.TRAIN_ONLY_SOURCE:
#         cfg.TRAIN.INPUT_SIZE_SOURCE = cfg.TRAIN.INPUT_SIZE_TARGET
#
#     # ------------------------------------------------------------------------
#     # create two yml files for oracle training with augmentation all and augmentation norm
#     # when you use oracel training set cfg.TRAIN_ONLY_SOURCE= True
#     # ------------------------------------------------------------------------
#
#     if cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL:
#         cfg.TRAIN.INPUT_SIZE_SOURCE = tuple(cfg.DATASET.CROP_SIZE)
#         cfg.TRAIN.INPUT_SIZE_TARGET = tuple(cfg.DATASET.CROP_SIZE)
#         cfg.TEST.INPUT_SIZE_TARGET = tuple(cfg.DATASET.CROP_SIZE)
#         cfg.TEST.OUTPUT_SIZE_TARGET = tuple(cfg.DATASET.CROP_SIZE)
#
#     if cfg.REPRODUCE_PANOPTIC_DEEPLAB and not cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL:
#         cfg.TRAIN.INPUT_SIZE_SOURCE = cfg.TEST.OUTPUT_SIZE_TARGET
#         cfg.TRAIN.INPUT_SIZE_TARGET = cfg.TEST.OUTPUT_SIZE_TARGET
#         cfg.TEST.INPUT_SIZE_TARGET = cfg.TEST.OUTPUT_SIZE_TARGET
#         cfg.TEST.OUTPUT_SIZE_TARGET = cfg.TEST.OUTPUT_SIZE_TARGET
#     # ------------------------------------------------------------------------
#
#
#     cfg.MACHINE = cmdline_inputs.machine
#
#     if cfg.MACHINE == 0: # DESKTOP
#         # EXT-HDD-0
#         machine_specific_exp_root_dir = '/media/suman/CVLHDD/apps/experiments/CVPR2022/cvpr2022/debug'
#         cfg.DATA_ROOT = '/media/suman/CVLHDD/apps/datasets'
#         # EXT-HDD-1
#         # machine_specific_exp_root_dir = '/media/suman/DATADISK2/apps'
#         # cfg.DATA_ROOT = '/media/suman/DATADISK2/apps/datasets'
#         pretrained_model_path = '/home/suman/apps/code/CVPR2021/MTI_Simon_ECCV2020/mti_simon/dada/pretrained_models'
#
#     elif cfg.MACHINE == 1: # DGX
#         machine_specific_exp_root_dir = '/raid/susaha/experiments/CVPR2022/cvpr2022'
#         cfg.DATA_ROOT = '/raid/susaha/datasets'
#         pretrained_model_path = '/raid/susaha/pretrained_models'
#
#     elif cfg.MACHINE == 2: # AWS
#         machine_specific_exp_root_dir = '/mnt/efs/fs1'
#         cfg.DATA_ROOT = '/mnt/efs/fs1/datasets'
#         pretrained_model_path = '/mnt/efs/fs1/cvpr_exp/pretrained_imagement'
#
#     elif cfg.MACHINE == 3: # Euler
#         machine_specific_exp_root_dir = '/cluster/work/cvl/susaha/experiments/CVPR2022/cvpr2022'
#         cfg.DATA_ROOT = cmdline_inputs.data_root  # on euler the data is unpacked and stored in a temporary directory for training, so this path is generated from the bash script based on $TMPDIR path
#         pretrained_model_path = cmdline_inputs.pret_model
#
#
#     if cfg.MACHINE == 0: # DESKTOP
#         # exp_root = datetime.now().strftime("%m-%Y")
#         # phase_name = datetime.now().strftime("%d-%m-%Y")
#         # sub_phase_name = datetime.now().strftime("%H-%M-%S-%f")
#         # cfg.EXP_ROOT = 'exproot_{}'.format(exp_root)
#         # cfg.EXP_PHASE = 'phase_{}'.format(phase_name)
#         # cfg.EXP_SUB_PHASE = 'subphase_{}'.format(sub_phase_name)
#         cfg.EXP_ROOT = 'exproot_debug'
#         cfg.EXP_PHASE = 'phase_debug'
#         cfg.EXP_SUB_PHASE = 'subphase_debug'
#
#     elif cfg.MACHINE == 1 or cfg.MACHINE == 2 or cfg.MACHINE == 3: # DGX or AWS or Euler
#         cfg.EXP_ROOT = 'exproot_{}'.format(cmdline_inputs.exp_root) # TODO: need to define in the commandline
#         cfg.EXP_PHASE = 'phase_{}'.format(cmdline_inputs.phase_name)    # TODO: need to define in the commandline
#         cfg.EXP_SUB_PHASE = 'subphase_{}'.format(cmdline_inputs.sub_phase_name) # TODO: need to define in the commandline
#         cfg.LOSS.CENTER.WEIGHT = cmdline_inputs.loss_center_weight
#         cfg.SOLVER.BASE_LR = cmdline_inputs.solver_base_lr
#         cfg.TEST.EVAL_INSTANCE=True if cmdline_inputs.eval_instance == 'True' else False
#         cfg.LOSS.OFFSET.WEIGHT = cmdline_inputs.loss_offset_weight
#
#         print()
#         print('***')
#         print('cfg.LOSS.OFFSET.WEIGHT: {}, cfg.TEST.EVAL_INSTANCE: {}, cfg.LOSS.CENTER.WEIGHT: {}, cfg.SOLVER.BASE_LR: {}'.
#               format(cfg.LOSS.OFFSET.WEIGHT, cfg.TEST.EVAL_INSTANCE, cfg.LOSS.CENTER.WEIGHT, cfg.SOLVER.BASE_LR))
#         print('***')
#         print()
#     else:
#         raise NotImplementedError('Input correct machine id !'
#                                   'Error messege from ctrl/utils/common_config_panop.py --> def setup_exp_params(...): ' )
#
#     cfg.TRAIN.RESTORE_FROM = None
#     os.makedirs('{}/{}'.format(machine_specific_exp_root_dir, cfg.EXP_ROOT), exist_ok=True)
#     cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET = \
#         '{}/DeepLab_resnet_pretrained_imagenet.pth'.format(pretrained_model_path)
#
#     if cfg.TARGET == 'Mapillary':
#         cfg.TRAIN.INFO_TARGET = 'ctrl/dataset/mapillary_list/info.json'.format(cfg.NUM_CLASSES)
#         cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'Mapillary-Vista')
#     else:
#         cfg.TRAIN.INFO_TARGET = 'ctrl/dataset/cityscapes_list/info{}class.json'.format(cfg.NUM_CLASSES)
#         if cfg.CITYSCAPES_DATALOADING_MODE == 'panoptic':
#             cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'cityscapes_4_panoptic_deeplab/cityscapes')
#         else:
#             cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'Cityscapes')
#
#     cfg.TEST.INFO_TARGET = cfg.TRAIN.INFO_TARGET
#
#     cfg.DATA_DIRECTORY_SOURCE = osp.join(cfg.DATA_ROOT, 'Synthia/RAND_CITYSCAPES')
#
#     cfg.TRAIN.SNAPSHOT_DIR = osp.join(machine_specific_exp_root_dir, cfg.EXP_ROOT, cfg.EXP_PHASE, 'checkpoints', cfg.EXP_SUB_PHASE)
#     os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
#
#     cfg.TRAIN.LOG_DIR = osp.join(machine_specific_exp_root_dir, cfg.EXP_ROOT, cfg.EXP_PHASE, 'checkpoints', cfg.EXP_SUB_PHASE, 'train_logs')
#     os.makedirs(cfg.TRAIN.LOG_DIR, exist_ok=True)
#
#     cfg.TRAIN_LOG_FNAME = os.path.join(cfg.TRAIN.LOG_DIR, 'train_log.txt')
#
#     cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL = osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'best_model')
#     os.makedirs(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL, exist_ok=True)
#
#     cfg.TRAIN.TENSORBOARD_LOGDIR = cfg.TRAIN.SNAPSHOT_DIR.replace('checkpoints', 'tensorboard')
#     os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
#
#     cfg.TEST.VISUAL_RESULTS_DIR = cfg.TRAIN.SNAPSHOT_DIR.replace('checkpoints', 'visual_results')
#     os.makedirs(cfg.TEST.VISUAL_RESULTS_DIR, exist_ok=True)
#
#     # if cfg.IS_ISL == 'true':
#     #     cfg.WEIGHT_INITIALIZATION.RESUME_FROM_SNAPSHOT_GIVEN = cmdline_inputs.model_path
#     #     cfg.TRAIN.PSEUDO_LABELS_DIR = cfg.TRAIN.SNAPSHOT_DIR.replace('checkpoints', 'pseudo_labels')
#     #     os.makedirs(cfg.TRAIN.PSEUDO_LABELS_DIR, exist_ok=True)
#     #     cfg.PSEUDO_LABELS_SUBDIR = 'labels_{}-{}'.format(phase_name, sub_phase_name)
#     #     if cfg.DEBUG:
#     #         cfg.GEN_PSEUDO_LABELS_EVERY = [11, 21, 31, 41, 51, 61, 71, 81, 91]
#     #     else:
#     #         cfg.GEN_PSEUDO_LABELS_EVERY = [10001, 20001, 30001, 40001, 50001, 60001, 70001, 80001, 90000]
#     return cfg

# DEBUG
# for name, param in model_params_saved.items():
#     print('name: {}  param.size: {}'.format(name, param.size()))
# print('------------------------')
# for name, param in model_params_current.items():
#     print('name: {}  param.size()'.format(name, param.size()))
# print()