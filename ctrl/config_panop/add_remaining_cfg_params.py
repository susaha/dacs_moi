import os
import os.path as osp


def add_remaining_params(cfg):
    print()
    os.makedirs('{}/{}'.format(cfg.MS_EXP_ROOT, cfg.EXP_ROOT), exist_ok=True)
    cfg.TRAIN.DADA_DEEPLAB_RESENT_PRETRAINED_IMAGENET = '{}/DeepLab_resnet_pretrained_imagenet.pth'.format(cfg.TRAIN.IMGNET_PRETRAINED_MODEL_PATH)

    if cfg.TARGET == 'Mapillary':
        # cfg.TRAIN.INFO_TARGET = 'ctrl/dataset/mapillary_list/info.json'.format(cfg.NUM_CLASSES)
        # cfg.TRAIN.INFO_TARGET = 'ctrl/dataset/cityscapes_list/info{}class.json'.format(cfg.NUM_CLASSES)
        # cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'Mapillary-Vistas-v1.2')
        cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'mapillary_on_euler')
    elif cfg.TARGET == 'Cityscapes':
        cfg.TRAIN.INFO_TARGET = 'ctrl/dataset/cityscapes_list/info{}class.json'.format(cfg.NUM_CLASSES)
        if cfg.CITYSCAPES_DATALOADING_MODE == 'panoptic':
            cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'cityscapes_4_panoptic_deeplab/cityscapes')
        else:
            # cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'Cityscapes')
            cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'cityscapes_4_panoptic_deeplab/cityscapes')
    elif cfg.TARGET == 'Armasuisse':
        cfg.DATA_DIRECTORY_TARGET = osp.join(cfg.DATA_ROOT, 'Armasuisse')
        cfg.TRAIN.INFO_TARGET = 'ctrl/dataset/cityscapes_list/info{}class.json'.format(cfg.NUM_CLASSES)

    # as the class id for cityscpaes and mapillary are same - we can use this to evaluate both mapillary and cityscpaes
    cfg.TEST.INFO_TARGET = 'ctrl/dataset/cityscapes_list/info{}class.json'.format(cfg.NUM_CLASSES)

    cfg.DATA_DIRECTORY_SOURCE = osp.join(cfg.DATA_ROOT, 'Synthia/RAND_CITYSCAPES')

    cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.MS_EXP_ROOT, cfg.EXP_ROOT, cfg.PHASE_NAME, cfg.SUB_PHASE_NAME, 'checkpoints')
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    cfg.TEST.SNAPSHOT_DIR = cfg.TRAIN.SNAPSHOT_DIR

    cfg.TRAIN.LOG_DIR = osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_logs')
    os.makedirs(cfg.TRAIN.LOG_DIR, exist_ok=True)

    cfg.TRAIN_LOG_FNAME = os.path.join(cfg.TRAIN.LOG_DIR, 'train_log.txt')

    cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL = osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'best_model')
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL, exist_ok=True)

    cfg.TRAIN.TENSORBOARD_LOGDIR = cfg.TRAIN.SNAPSHOT_DIR.replace('checkpoints', 'tensorboard')
    os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)

    cfg.TEST.VISUAL_RESULTS_DIR = cfg.TRAIN.SNAPSHOT_DIR.replace('checkpoints', 'visual_results')
    os.makedirs(cfg.TEST.VISUAL_RESULTS_DIR, exist_ok=True)

    cfg.TRAIN.DACS_VISUAL_RESULTS_DIR = cfg.TRAIN.SNAPSHOT_DIR.replace('checkpoints', 'dacs_visual_results')
    os.makedirs(cfg.TRAIN.DACS_VISUAL_RESULTS_DIR, exist_ok=True)

    return cfg