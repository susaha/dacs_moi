import numpy as np
from ctrl.utils.serialization import json_load
from ctrl.dataset.base_dataset import BaseDataset
import os
import cv2
from PIL import Image
import torch
from ctrl.dataset.gen_panoptic_gt_labels import PanopticTargetGenerator, SemanticTargetGenerator
import logging
from ctrl.transforms_panop import build_transforms
from PIL import Image, ImageOps
from ctrl.dacs_old.data.build_aug import dacs_build_aug
from ctrl.dacs.data.augmentations import *

'''
the following cityscapes class to training id mapping is defined in ctrl/dataset/cityscapes_list/info16class.json
i got the respective class names from:  https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
(NOTE: 255 is background class and not cvonsidered during loss computation)
  "classes":16,
  "label2train":[
    [0, 255],
    [1, 255],
    [2, 255],
    [3, 255],
    [4, 255],
    [5, 255],
    [6, 255],
    [7, 0],     # road
    [8, 1],     # sidewalk
    [9, 255],   
    [10, 255],
    [11, 2],    # building
    [12, 3],    # wall
    [13, 4],    # fence 
    [14, 255],  # 
    [15, 255],
    [16, 255],
    [17, 5],    # pole
    [18, 255],  
    [19, 6],    # traffic light
    [20, 7],    # traffic sign
    [21, 8],    # vegetation
    [22, 255],  
    [23, 9],    # sky
    [24, 10],   # person
    [25, 11],   # rider
    [26, 12],   # car
    [27, 255],  
    [28, 13],   # bus
    [29, 255],
    [30, 255],  
    [31, 255],
    [32, 14],   # motorcycle
    [33, 15],   # bicycle
    [-1, 255]],

'''


# Add 1 void label.
# this for synthia-to-cityscapes 16 cls
_CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID_16 = [
    7,  # road
    8,  # sidewalk
    11, # building
    12, # wall
    13, # fence
    17, # pole
    19, # traffic light
    20, # traffic sign
    21, # vegetation
    23, # sky
    24, # person
    25, # rider
    26, # car
    28, # bus
    32, # motorcycle
    33, # bicycle
    0]
_CITYSCAPES_THING_LIST_16 = [10, 11, 12, 13, 14, 15] # # continuous trainIds

_CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID_19 = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0] # ids

_CITYSCAPES_THING_LIST_19 = [11, 12, 13, 14, 15, 16, 17, 18]  # continuous trainIds

class CityscapesDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val', max_iters=None, crop_size=(321, 321),
                 mean=(128, 128, 128), load_labels=True, info_path=None, labels_size=None,
                 transform=None, joint_transform=None, cfg=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean, joint_transform, cfg)

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/dataset/cityscapes_panop.py --> class CityscapesDataSet --> __init__() +++')
        self.logger.info('self.cfg.IS_ISL_TRAINING: {}'.format(self.cfg.IS_ISL_TRAINING))

        self.cfg = cfg
        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        self.joint_transform = joint_transform

        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

        cfgp = cfg['PANOPTIC_TARGET_GENERATOR']
        self.ignore_label = cfgp['IGNORE_LABEL']
        self.label_divisor = cfgp['LABEL_DIVISOR']
        if cfg.NUM_CLASSES == 16:
            self.cityscapes_thing_list = _CITYSCAPES_THING_LIST_16
        # elif cfg.NUM_CLASSES == 19:
        #     self.cityscapes_thing_list = _CITYSCAPES_THING_LIST_19
        #     print('*** self.cityscapes_thing_list: {} ***'.format(self.cityscapes_thing_list))

        # self.ignore_stuff_in_offset = cfgp['IGNORE_STUFF_IN_OFFSET']
        # self.small_instance_area = cfgp['SMALL_INSTANCE_AREA']
        # self.small_instance_weight = cfgp['SMALL_INSTANCE_WEIGHT']
        # self.sigma = cfgp['SIGMA']
        # self.target_transform = PanopticTargetGenerator(self.logger, self.ignore_label, self.rgb2id, self.cityscapes_thing_list,
        #                                                 sigma=8, ignore_stuff_in_offset=self.ignore_stuff_in_offset,
        #                                                 small_instance_area=self.small_instance_area,
        #                                                 small_instance_weight=self.small_instance_weight)

        # if self.set == 'val':
        #     self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)

        if self.set == 'val':
            self.input_size = (1024, 2048)
        else:
            self.input_size = (self.cfg.DATASET.RANDOM_CROP_DIM, self.cfg.DATASET.RANDOM_CROP_DIM)

        self.augmentations = Compose([RandomCrop_city(self.input_size)])
        self.is_transform = True
        self.interpolation = Image.BILINEAR

    def get_metadata(self, name, mode=None):
        img_file = self.root / 'leftImg8bit' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):

        img_file, label_file, name = self.files[index]

        img = Image.open(img_file)
        img = np.array(img, dtype=np.uint8)

        lbl = Image.open(label_file)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.map_labels(lbl)

        if self.set != 'val':
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, np.array(img.shape), name


    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        # img = m.imresize(img, (self.input_size[0], self.input_size[1]))  # uint8 with RGB mode
        img = np.array(Image.fromarray(img).resize((self.input_size[0], self.input_size[1]), self.interpolation)) # (width, height)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)

        # lbl = m.imresize(lbl, (self.input_size[0], self.input_size[1]), "nearest", mode="F")
        lbl = np.array(Image.fromarray(lbl).resize((self.input_size[0], self.input_size[1]), Image.NEAREST)) # (width, height)

        lbl = lbl.astype(int)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        # if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
        #     print("after det", classes, np.unique(lbl))
        #     raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    # this function is copied from:
    # /home/suman/apps/code/CVPR2022/panoptic-deeplab/segmentation/data/datasets/base_dataset.py
    @staticmethod
    def read_image(file_name, format=None, do_resize=None, size=None, interpolation=None):
        image = Image.open(file_name)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)

        if do_resize:
            image = image.resize(size, interpolation)

        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image

    # this function is copied from:
    # /home/suman/apps/code/CVPR2022/panoptic-deeplab/segmentation/data/datasets/cityscapes_panoptic.py
    @staticmethod
    def train_id_to_eval_id(num_classes):
        if num_classes == 16:
            return _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID_16
        elif num_classes == 19:
            return _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID_19


    # this function is copied from:
    # /home/suman/apps/code/CVPR2022/panoptic-deeplab/segmentation/data/datasets/cityscapes_panoptic.py
    @staticmethod
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    @staticmethod
    def create_label_colormap(num_classes):
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        if num_classes == 16:
            colormap[0] = [128, 64, 128]  # road; 804080ff
            colormap[1] = [244, 35, 232]  # sidewalk; f423e8ff
            colormap[2] = [70, 70, 70]  # building; 464646ff
            colormap[3] = [102, 102, 156]  # wall; 666699ff
            colormap[4] = [190, 153, 153]  # fence; be9999ff
            colormap[5] = [153, 153, 153]  # pole; 999999ff
            colormap[6] = [250, 170, 30]  # traffic-light; faaa1eff
            colormap[7] = [220, 220, 0]  # traffic-sign; dcdc00ff
            colormap[8] = [107, 142, 35]  # vegetation; 6b8e23ff
            colormap[9] = [70, 130, 180]  # sky; 4682b4ff
            colormap[10] = [220, 20, 60]  # person; dc143cff
            colormap[11] = [255, 0, 0]  # rider; ff0000ff
            colormap[12] = [0, 0, 142]  # car; 00008eff
            colormap[13] = [0, 60, 100]  # bus; 003c64ff
            colormap[14] = [0, 0, 230]  # motocycle, 0000e6ff
            colormap[15] = [119, 11, 32]  # bicycle, 770b20ff

        elif num_classes == 19:
            colormap[0] = [128, 64, 128]
            colormap[1] = [244, 35, 232]
            colormap[2] = [70, 70, 70]
            colormap[3] = [102, 102, 156]
            colormap[4] = [190, 153, 153]
            colormap[5] = [153, 153, 153]
            colormap[6] = [250, 170, 30]
            colormap[7] = [220, 220, 0]
            colormap[8] = [107, 142, 35]
            colormap[9] = [152, 251, 152]
            colormap[10] = [70, 130, 180]
            colormap[11] = [220, 20, 60]
            colormap[12] = [255, 0, 0]
            colormap[13] = [0, 0, 142]
            colormap[14] = [0, 0, 70]
            colormap[15] = [0, 60, 100]
            colormap[16] = [0, 80, 100]
            colormap[17] = [0, 0, 230]
            colormap[18] = [119, 11, 32]

        return colormap