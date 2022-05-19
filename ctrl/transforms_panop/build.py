# ------------------------------------------------------------------------------
# Builds transformation for both image and labels.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

# from . import transforms as T # original
from . import transforms_v2 as T
import logging


def build_transforms(transform_param, is_train=True, mode=0, dataset=None):
    logger = logging.getLogger(__name__)
    logger.info('ctrl/transforms_panop/build.py --> applying transforms: {}'.format(dataset))
    if is_train:
        min_scale = transform_param['min_scale']
        max_scale = transform_param['max_scale']
        scale_step_size = transform_param['scale_step_size']
        crop_h = transform_param['crop_h']
        crop_w = transform_param['crop_w']
        pad_value = transform_param['pad_value']
        ignore_label = transform_param['ignore_label']
        flip_prob = transform_param['flip_prob']
        mean = transform_param['mean']
        std = transform_param['std']
    else:
        # no data augmentation
        min_scale = 1
        max_scale = 1
        scale_step_size = 0
        flip_prob = 0
        crop_h = transform_param['crop_h']
        crop_w = transform_param['crop_w']
        pad_value = transform_param['pad_value']
        ignore_label = transform_param['ignore_label']
        mean = transform_param['mean']
        std = transform_param['std']

    if mode == 0:
        transforms = T.Compose(
            [
                T.RandomScale(
                    min_scale,
                    max_scale,
                    scale_step_size
                ),
                T.RandomCrop(
                    crop_h,
                    crop_w,
                    pad_value,
                    ignore_label,
                    random_pad=is_train
                ),
                T.RandomHorizontalFlip(flip_prob),
                T.ToTensor(),
                T.Normalize(
                    mean,
                    std
                )
            ]
        )
    elif mode == 1:
        transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean,
                    std
                )
            ]
        )
    elif mode == 2:
        transforms = T.RandomCrop(
                    crop_h,
                    crop_w,
                    pad_value,
                    ignore_label,
                    random_pad=is_train
                )
    else:
        raise NotImplementedError('ctrl/transforms_panop/build.py  --> mode value not defined!!')


    return transforms
