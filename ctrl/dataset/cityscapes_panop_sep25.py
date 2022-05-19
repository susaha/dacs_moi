import numpy as np
from ctrl.utils.serialization import json_load
from ctrl.dataset.base_dataset_sep25 import BaseDataset
# from ctrl.dataset.base_dataset import BaseDataset
import os
import cv2
from PIL import Image
import torch
from ctrl.dataset.gen_panoptic_gt_labels_sep25 import PanopticTargetGenerator, SemanticTargetGenerator
# from ctrl.dataset.gen_panoptic_gt_labels import PanopticTargetGenerator, SemanticTargetGenerator
import logging
from ctrl.transforms_panop import build_transforms
from PIL import Image, ImageOps


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
        elif cfg.NUM_CLASSES == 19:
            self.cityscapes_thing_list = _CITYSCAPES_THING_LIST_19
            print('*** self.cityscapes_thing_list: {} ***'.format(self.cityscapes_thing_list))

        self.ignore_stuff_in_offset = cfgp['IGNORE_STUFF_IN_OFFSET']
        self.small_instance_area = cfgp['SMALL_INSTANCE_AREA']
        self.small_instance_weight = cfgp['SMALL_INSTANCE_WEIGHT']
        self.sigma = cfgp['SIGMA']

        transform_param = {}
        if self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL or self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ONLY_NORM:
            transform_param['min_scale'] = cfg.DATASET.MIN_SCALE
            transform_param['max_scale'] = cfg.DATASET.MAX_SCALE
            transform_param['scale_step_size'] = cfg.DATASET.SCALE_STEP_SIZE
            transform_param['crop_h'] = cfg.DATASET.CROP_SIZE[1]
            transform_param['crop_w'] = cfg.DATASET.CROP_SIZE[0]
            transform_param['pad_value'] = tuple([int(v * 255) for v in tuple(self.cfg.DATASET.MEAN)])
            self.label_pad_value = (0, 0, 0)
            transform_param['ignore_label'] = self.label_pad_value
            transform_param['flip_prob'] = 0.5 if self.cfg.DATASET.MIRROR else 0
            transform_param['mean'] = self.cfg.DATASET.MEAN
            transform_param['std'] = self.cfg.DATASET.STD

        elif self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_RANDOM_CROP:
            if self.set == 'train':
                transform_param['crop_h'] = self.cfg.DATASET.RANDOM_CROP_DIM
                transform_param['crop_w'] = self.cfg.DATASET.RANDOM_CROP_DIM
            else:
                transform_param['crop_h'] = self.cfg.TEST.INPUT_SIZE_TARGET[1]
                transform_param['crop_w'] = self.cfg.TEST.INPUT_SIZE_TARGET[0]
            transform_param['pad_value'] = tuple([int(v * 255) for v in tuple(self.cfg.TRAIN.IMG_MEAN)])
            self.label_pad_value = (0, 0, 0)
            transform_param['ignore_label'] = self.label_pad_value
            transform_param['flip_prob'] = 0.0
            transform_param['mean'] = []
            transform_param['std'] = []
            transform_param['min_scale'] = None
            transform_param['max_scale'] = None
            transform_param['scale_step_size'] = None

        if self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL or self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ONLY_NORM or self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_RANDOM_CROP:
            mode = None
            if self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL == True:
                mode = 0
            elif self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ONLY_NORM == True:
                mode = 1
            elif self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_RANDOM_CROP == True:
                mode = 2
            if self.set == 'train':
                self.is_train = True
            else:
                self.is_train = False
            self.transform = build_transforms(transform_param, is_train=self.is_train, mode=mode, dataset='cityscapes')

        self.target_transform = PanopticTargetGenerator(self.logger, self.ignore_label, self.rgb2id, self.cityscapes_thing_list,
                                                        sigma=8, ignore_stuff_in_offset=self.ignore_stuff_in_offset,
                                                        small_instance_area=self.small_instance_area,
                                                        small_instance_weight=self.small_instance_weight)

        if self.set == 'val':
            self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)

    def get_metadata(self, name, mode=None):
        print('for data loading type panoptic, the paths are generated from the json file, refer to to base_dataset.py')
        # img_file = self.root / 'leftImg8bit_trainvaltest/leftImg8bit' / self.set / name
        # if self.cfg.IS_ISL_TRAINING and self.set == 'train':
        #     str1 = name.split('/')
        #     str2 = str1[1].split('.')
        #     str3 = os.path.join(str1[0], str2[0] + '.npy')
        #     # arxiv version
        #     label_file = os.path.join(self.cfg.TRAIN.PSEUDO_LABELS_DIR, 'cityscapes', 'train', self.cfg.PSEUDO_LABELS_SUBDIR,
        #                               'nparrays_{:.1f}'.format(self.cfg.ISL_THRESHOLD), str3)
        #     # cvpr submitted version
        #     # label_file = os.path.join(self.root, 'gtFinePseudo_trainvaltest/gtFine',
        #     #                          self.set, self.cfg.MODEL_FOR_SUDO_LABEL_TRAINING,
        #     #                          'nparrays_{:.1f}'.format(self.cfg.ISL_THRESHOLD), str3)
        # else:
        #     label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        #     label_file = self.root / 'gtFine_trainvaltest/gtFine' / self.set / label_name
        # label_name = name.replace("leftImg8bit", "disparity")
        # disp_img_file = self.root / 'disparity_trainvaltest/disparity' / self.set / label_name
        # label_name = name.replace("leftImg8bit", "camera")
        # label_name = label_name.replace("png", "json")
        # disp_json_file = self.root / 'camera_trainvaltest/camera' / self.set / label_name
        # return img_file, label_file, disp_img_file, disp_json_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        panop_label_file = None
        segment_info = None
        raw_label = None
        if self.dataloading_mode == 'panoptic':
            img_file, label_file, panop_label_file, disp_file, calib_file, segment_info, name = self.files[index]
        else:
            img_file, label_file,  disp_file, calib_file, name = self.files[index]

        if self.cfg.IS_ISL_TRAINING and self.set == 'train':
            with open(label_file, 'rb') as f:
                semseg_label = np.load(f)
                if self.cfg.DEBUG:
                    semseg_label = cv2.resize(semseg_label, tuple(self.labels_size), interpolation=cv2.INTER_NEAREST)
        else:
            semseg_label = self.get_labels(label_file)

        if self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL:
            image = self.read_image(img_file, 'RGB')
        elif self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ONLY_NORM:
            image = self.read_image(img_file, 'RGB', do_resize=True, size=self.labels_size, interpolation=Image.NEAREST)
        else:
            image = self.get_image(img_file)
            image = self.preprocess(image)
            if self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_RANDOM_CROP:
                image = image.transpose((1, 2, 0))

        depth_labels = self.get_depth_labels(disp_file, calib_file)

        if self.cfg.IS_ISL_TRAINING and self.set == 'train':
            pass
        else:
            semseg_label = self.map_labels(semseg_label).copy()

        # read panoptic targets or ground-truths
        if self.dataloading_mode == 'panoptic':
            label_panop = Image.open(panop_label_file)

            if not self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL:
                label_panop = label_panop.resize(self.labels_size, Image.NEAREST)
            label_panop = np.asarray(label_panop, dtype=np.float32)  # the id values are > 255, we need np.float32

            if self.set == 'val':
                raw_label = label_panop.copy()
                raw_label = self.raw_label_transform(raw_label, segment_info)['semantic']

            if self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL or self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ONLY_NORM or self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_RANDOM_CROP:
                image, label_panop, semseg_label = self.transform(image, label_panop, semseg_label, train_mode=self.is_train)
                if self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_RANDOM_CROP:
                    image = image.transpose((2, 0, 1))

            label_panop_dict = self.target_transform(label_panop, segment_info, VIS=False)

            label_panop_dict['semantic'] = torch.as_tensor(semseg_label.astype('long'))  # semanitc gt label
            label_panop_dict['depth'] = depth_labels.copy()

            if self.set == 'val':
                label_panop_dict['raw_label'] = raw_label

            if self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL or self.cfg.PANOPTIC_DEEPLAB_DATA_AUG_ONLY_NORM:
                return image, label_panop_dict, np.array(image.shape), name
            else:
                return image.copy(), label_panop_dict, np.array(image.shape), name
        else:
            return image.copy(), semseg_label, depth_labels.copy(), np.array(image.shape), name

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