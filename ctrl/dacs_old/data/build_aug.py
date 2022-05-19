# ------------------------------------------------------------------------------
# Builds transformation for both image and labels.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

# from . import transforms as T # original
from . import augmentation as T
import logging


def dacs_build_augV6(transform_param, is_train=True, dataset=None):
    logger = logging.getLogger(__name__)
    logger.info('ctrl/dacs_old/data/build_aug.py --> dacs_build_augV6() applying transforms: {}'.format(dataset))
    if is_train:
        crop_h = transform_param['crop_h']
        crop_w = transform_param['crop_w']
        transforms = T.RandomCropDACSV6(
                    crop_h,
                    crop_w,
                    dataset=dataset
                )
    else:
        raise NotImplementedError('ctrl/dacs_old/data/build_aug.py  --> In DACS paper, random crop is applied during training only !!')
    return transforms

def dacs_build_augV5(transform_param, is_train=True, dataset=None):
    logger = logging.getLogger(__name__)
    logger.info('ctrl/dacs_old/data/build_aug.py --> dacs_build_augV5() applying transforms: {}'.format(dataset))
    if is_train:
        crop_h = transform_param['crop_h']
        crop_w = transform_param['crop_w']
        transforms = T.RandomCropDACSV5(
                    crop_h,
                    crop_w,
                    dataset=dataset
                )
    else:
        raise NotImplementedError('ctrl/dacs_old/data/build_aug.py  --> In DACS paper, random crop is applied during training only !!')
    return transforms

def dacs_build_augV4(transform_param, is_train=True, dataset=None):
    logger = logging.getLogger(__name__)
    logger.info('ctrl/dacs_old/data/build_aug.py --> dacs_build_augV4() applying transforms: {}'.format(dataset))
    if is_train:
        crop_h = transform_param['crop_h']
        crop_w = transform_param['crop_w']
        transforms = T.RandomCropDACSV4(
                    crop_h,
                    crop_w,
                    dataset=dataset
                )
    else:
        raise NotImplementedError('ctrl/dacs_old/data/build_aug.py  --> In DACS paper, random crop is applied during training only !!')
    return transforms

def dacs_build_augV3(transform_param, is_train=True, dataset=None):
    logger = logging.getLogger(__name__)
    logger.info('ctrl/dacs_old/data/build_aug.py --> dacs_build_augV2() applying transforms: {}'.format(dataset))
    if is_train:
        crop_h = transform_param['crop_h']
        crop_w = transform_param['crop_w']
        transforms = T.RandomCropDACSV3(
                    crop_h,
                    crop_w,
                    dataset=dataset
                )
    else:
        raise NotImplementedError('ctrl/dacs_old/data/build_aug.py  --> In DACS paper, random crop is applied during training only !!')
    return transforms

def dacs_build_augV2(transform_param, is_train=True, dataset=None):
    logger = logging.getLogger(__name__)
    logger.info('ctrl/dacs_old/data/build_aug.py --> dacs_build_augV2() applying transforms: {}'.format(dataset))
    if is_train:
        crop_h = transform_param['crop_h']
        crop_w = transform_param['crop_w']
        transforms = T.RandomCropDACSV2(
                    crop_h,
                    crop_w,
                    dataset=dataset
                )
    else:
        raise NotImplementedError('ctrl/dacs_old/data/build_aug.py  --> In DACS paper, random crop is applied during training only !!')
    return transforms


def dacs_build_aug(transform_param, is_train=True, dataset=None):
    logger = logging.getLogger(__name__)
    logger.info('ctrl/dacs_old/data/build_aug.py --> applying transforms: {}'.format(dataset))
    if is_train:
        crop_h = transform_param['crop_h']
        crop_w = transform_param['crop_w']
        transforms = T.RandomCropDACS(
                    crop_h,
                    crop_w,
                    dataset=dataset
                )
    else:
        raise NotImplementedError('ctrl/dacs_old/data/build_aug.py  --> In DACS paper, random crop is applied during training only !!')
    return transforms
