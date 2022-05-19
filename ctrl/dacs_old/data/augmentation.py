import random
import cv2
import numpy as np
from torchvision.transforms import functional as F
import logging


import random
import cv2
import numpy as np
from torchvision.transforms import functional as F
import logging

class Compose(object):
    """
    Composes a sequence of transforms.
    Arguments:
        transforms: A list of transforms.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, label2):
        for t in self.transforms:
            image, label, label2 = t(image, label, label2)
        return image, label, label2

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomCropDACSV6(object):
    def __init__(self, crop_h, crop_w, dataset='cityscapes'):
        logger = logging.getLogger(__name__)
        logger.info('ctrl/dacs_old/data/augmentation.py --> class RandomCropDACSV6(...) : def __init__(...)')

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.dataset = dataset

    def __call__(self, image, label2, train_mode=None, new_image_shape=None):
        image_dtype = image.dtype
        label2_dtype = label2.dtype
        img_pad, label2_pad = image, label2

        if train_mode:
            # img_h_target, img_w_target = img_pad.shape[0], img_pad.shape[1]  # this might crop from the padded regions, as the image might have some padded regions as well in mapillary
            img_w, img_h = new_image_shape  # this limit the random crop to crop from regions where the actual image is present and exculde padded regions

            # if the actual mapillary image region (exculding the padded region) is smaller than the cropp size then allow the crop to be done in the padded region too
            if img_w < self.crop_w:
                img_w = int(self.crop_w)
            if img_h < self.crop_h:
                img_h = int(self.crop_h)

            h_off = random.randint(0, int(img_h) - self.crop_h)
            w_off = random.randint(0, int(img_w) - self.crop_w)
            image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            label2 = np.asarray(label2_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            return image.astype(image_dtype), label2.astype(label2_dtype)
        else:
            return image, label2


class RandomCropDACSV5(object):
    def __init__(self, crop_h, crop_w, dataset='cityscapes'):
        logger = logging.getLogger(__name__)
        logger.info('ctrl/dacs_old/data/augmentation.py --> class RandomCropDACSV5(...) : def __init__(...)')

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.dataset = dataset

    def __call__(self, image, label2, depth=None, use_depth=False, train_mode=None, new_image_shape=None):
        image_dtype = image.dtype
        label2_dtype = label2.dtype
        if use_depth:
            depth_dtype = depth.dtype
        else:
            depth = None

        img_pad, label2_pad, depth_pad = image, label2, depth

        if train_mode:
            img_h_target, img_w_target = img_pad.shape[0], img_pad.shape[1] # this might crop from the padded regions, as the image might have some padded regions as well in mapillary
            img_w, img_h = new_image_shape # this limit the random crop to crop from regions where the actual image is present and exculde padded regions

            # if the actual mapillary image region (exculding the padded region) is smaller than the cropp size then allow the crop to be done in the padded region too
            if img_w < self.crop_w:
                img_w = int(img_w_target)
            if img_h < self.crop_h:
                img_h = int(img_h_target)

            h_off = random.randint(0, int(img_h) - self.crop_h)
            w_off = random.randint(0, int(img_w) - self.crop_w)

            image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            label2 = np.asarray(label2_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            if use_depth:
                depth = np.asarray(depth_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
                depth = depth.astype(depth_dtype)

            return image.astype(image_dtype), label2.astype(label2_dtype), depth
        else:
            return image, label2


class RandomCropDACSV4(object):
    def __init__(self, crop_h, crop_w, dataset='cityscapes'):
        logger = logging.getLogger(__name__)
        logger.info('ctrl/dacs_old/data/augmentation.py --> class RandomCropDACSV4(...) : def __init__(...)')

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.dataset = dataset

    def __call__(self, image, label2, depth=None, use_depth=False, train_mode=None):
        image_dtype = image.dtype
        label2_dtype = label2.dtype
        if use_depth:
            depth_dtype = depth.dtype
        else:
            depth = None

        img_pad, label2_pad, depth_pad = image, label2, depth

        if train_mode:
            img_h, img_w = img_pad.shape[0], img_pad.shape[1]
            h_off = random.randint(0, int(img_h) - self.crop_h)
            w_off = random.randint(0, int(img_w) - self.crop_w)

            image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            label2 = np.asarray(label2_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            if use_depth:
                depth = np.asarray(depth_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
                depth = depth.astype(depth_dtype)

            return image.astype(image_dtype), label2.astype(label2_dtype), depth
        else:
            return image, label2


class RandomCropDACSV2(object):
    def __init__(self, crop_h, crop_w, dataset='cityscapes'):
        logger = logging.getLogger(__name__)
        logger.info('ctrl/dacs_old/data/augmentation.py --> class RandomCropDACSV2(...) : def __init__(...)')

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.dataset = dataset

    def __call__(self, image, label2, center, center_w, offset, offset_w, depth=None, use_depth=False, train_mode=None):
        image_dtype = image.dtype
        label2_dtype = label2.dtype
        center_dtype = center.dtype
        center_w_dtype = center_w.dtype
        offset_dtype = offset.dtype
        offset_w_dtype = offset_w.dtype
        if use_depth:
            depth_dtype = depth.dtype
        else:
            depth = None

        img_pad, label2_pad, center_pad, center_w_pad, offset_pad, offset_w_pad, depth_pad = image, label2, center, center_w, offset, offset_w, depth

        if train_mode:
            img_h, img_w = img_pad.shape[0], img_pad.shape[1]
            h_off = random.randint(0, int(img_h) - self.crop_h)
            w_off = random.randint(0, int(img_w) - self.crop_w)

            image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            label2 = np.asarray(label2_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            center = np.asarray(center_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            center_w = np.asarray(center_w_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            offset = np.asarray(offset_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            offset_w = np.asarray(offset_w_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            if use_depth:
                depth = np.asarray(depth_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
                depth = depth.astype(depth_dtype)

            return image.astype(image_dtype), label2.astype(label2_dtype), \
                   center.astype(center_dtype), center_w.astype(center_w_dtype), \
                   offset.astype(offset_dtype), offset_w.astype(offset_w_dtype), depth

        else:
            return image, label2


class RandomCropDACSV3(object):
    def __init__(self, crop_h, crop_w, dataset='cityscapes'):
        logger = logging.getLogger(__name__)
        logger.info('ctrl/dacs_old/data/augmentation.py --> class RandomCropDACSV3(...) : def __init__(...)')

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.dataset = dataset

    def __call__(self, image, label2, depth=None, use_depth=False, train_mode=None):
        image_dtype = image.dtype
        label2_dtype = label2.dtype
        if use_depth:
            depth_dtype = depth.dtype
        else:
            depth = None
        img_pad, label2_pad, depth_pad = image, label2, depth

        if train_mode:
            img_h, img_w = img_pad.shape[0], img_pad.shape[1]
            h_off = random.randint(0, int(img_h) - self.crop_h)
            w_off = random.randint(0, int(img_w) - self.crop_w)
            image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            label2 = np.asarray(label2_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            if use_depth:
                depth = np.asarray(depth_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
                depth = depth.astype(depth_dtype)
            return image.astype(image_dtype), label2.astype(label2_dtype), depth
        else:
            return image, label2


class RandomCropDACS(object):
    def __init__(self, crop_h, crop_w, dataset='cityscapes'):
        logger = logging.getLogger(__name__)
        logger.info('ctrl/dacs_old/data/augmentation.py --> class RandomCropDACS(...) : def __init__(...)')

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.dataset = dataset

    def __call__(self, image, label2, train_mode=None):
        image_dtype = image.dtype
        label2_dtype = label2.dtype
        img_pad, label2_pad = image, label2
        if train_mode:
            img_h, img_w = img_pad.shape[0], img_pad.shape[1]
            h_off = random.randint(0, int(img_h) - self.crop_h)
            w_off = random.randint(0, int(img_w) - self.crop_w)
            image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            label2 = np.asarray(label2_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            return image.astype(image_dtype), label2.astype(label2_dtype)
        else:
            return image, label2


'''
    def __call__(self, image, label=None, label2=None, label_depth=None, crop_depth=None, train_mode=None):

    if self.dataset == 'mapillary':
        raise NotImplementedError('For Mapillary, there is no implementation found!')

    img_h, img_w = image.shape[0], image.shape[1]

    # save dtype
    image_dtype = image.dtype
    if label:
        label_dtype = label.dtype
    label2_dtype = label2.dtype
    label_depth_dtype = None
    if crop_depth:
        label_depth_dtype = label_depth.dtype

    label_depth_pad = None

    if crop_depth:
        img_pad, label_pad, label2_pad, label_depth_pad = image, label, label2, label_depth
    else:
        img_pad, label_pad, label2_pad = image, label, label2

    if train_mode:
        img_h, img_w = img_pad.shape[0], img_pad.shape[1]

        # if self.dataset == 'cityscapes':
        #     h_off = random.randint(0, int(img_h/2) - self.crop_h)
        #     w_off = random.randint(0, int(img_w/2) - self.crop_w)
        # else:
        h_off = random.randint(0, int(img_h) - self.crop_h)
        w_off = random.randint(0, int(img_w) - self.crop_w)

        image = np.asarray(img_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
        label2 = np.asarray(label2_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
        if crop_depth:
            label_depth = np.asarray(label_depth_pad[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w], np.float32)
            return image.astype(image_dtype), label.astype(label_dtype), label2.astype(label2_dtype), label_depth.astype(label_depth_dtype)
        else:
            return image.astype(image_dtype), label.astype(label_dtype), label2.astype(label2_dtype)
    else:
        if crop_depth:
            return image, label, label2, label_depth
        else:
            return image, label, label2
'''