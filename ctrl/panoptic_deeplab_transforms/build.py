
# ------------------------------------------------------------------------------
# Builds transformation for both image and labels.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from . import transforms as T


def build_transforms(is_train=True, min_scale=0.5, max_scale=2.0, scale_step_size=0.1,
                     crop_h=512, crop_w=512, pad_value=(123, 116, 103),
                     ignore_label=0, flip_prob=0.5, use_depth=None, depth=None):   # in panoptic deeplab, for semantic label, ignore_label = 255, for panoptic label we use 0
    if is_train:
        min_scale = min_scale
        max_scale = max_scale
        scale_step_size = scale_step_size
        crop_h = crop_h
        crop_w = crop_w
        pad_value = pad_value
        ignore_label = ignore_label
        flip_prob = flip_prob

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
            T.RandomHorizontalFlip(flip_prob)
        ]
    )
    return transforms
