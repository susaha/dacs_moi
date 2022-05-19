import numpy as np
from ctrl.utils.serialization import json_load
from ctrl.dataset.base_dataset_sep25 import BaseDataset
# from ctrl.dataset.base_dataset import BaseDataset
import os
import cv2
from PIL import Image
import torch
# from ctrl.dataset.gen_panoptic_gt_labels_sep25 import PanopticTargetGenerator, SemanticTargetGenerator
from ctrl.dataset.gen_panoptic_gt_labels_v2 import PanopticTargetGenerator, SemanticTargetGenerator
import logging
from ctrl.transforms_panop import build_transforms
from PIL import Image, ImageOps
from torch.utils import data
import json
# from ctrl.dacs_old.data.build_aug import dacs_build_aug
from ctrl.dacs_old.data.build_aug import dacs_build_aug, dacs_build_augV2, dacs_build_augV4
from ctrl.utils.synthia_cityscapes_gt_visualization import visualize_panoptic_gt, visualize_panoptic_gt_calss_wise
from ctrl.utils.panoptic_deeplab.save_annotation import save_heatmap_image, save_offset_image_v2, save_annotation


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

'''
/media/suman/CVLHDD/apps/datasets/Armasuisse$ ls -l
    total 596
    drwxrwxr-x  2 suman suman   4096 Feb 21 15:59 annotations            
    drwxrwxr-x  2 suman suman 364544 Feb 21 15:31 resized_imgs_train
    drwxrwxr-x  2 suman suman 237568 Feb 21 16:03 resized_imgs_val           
    .
    ├── frame_list_train.txt
    ├── frame_list_val.txt

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

# _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID_19 = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0] # ids
# _CITYSCAPES_THING_LIST_19 = [11, 12, 13, 14, 15, 16, 17, 18]  # continuous trainIds

class ArmasuisseDataSet(data.Dataset):

    def __init__(self, root, list_path, set='val', max_iters=None, crop_size=(512, 1024),
                 mean=(128, 128, 128), load_labels=True, info_path=None, labels_size=(1024, 2048),
                 transform=None, joint_transform=None, cfg=None, gen_panop_labels=False, file_range=None):

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/dataset/armasuisse_panop_jan04_2022_v2.py --> class ArmasuisseDataSet --> __init__() +++')
        self.gen_panop_labels = gen_panop_labels
        self.root = root
        self.cfg = cfg
        self.set = set
        self.image_size = crop_size
        self.labels_size = labels_size
        self.mean = mean
        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=str)
        self.cityscapes_thing_list = _CITYSCAPES_THING_LIST_16
        print('*** self.cityscapes_thing_list: {} ***'.format(self.cityscapes_thing_list))
        self.files = []
        # self.root = '/media/suman/CVLHDD/apps/datasets/Armasuisse'
        if self.set == 'train' or self.set == 'val':
            frm_list_fname = os.path.join(self.root, 'annotations', 'frame_list_{}.txt'.format(self.set))
            file1 = open(frm_list_fname, 'r')
            while True:
                img_fname = file1.readline().strip()
                if not img_fname:
                    break
                img_file = os.path.join(self.root, 'resized_imgs_{}'.format(self.set), img_fname)
                self.files.append((img_file, img_fname))

        elif self.set == 'all':
            frm_list_fname = os.path.join(self.root, 'annotations', 'frame_list_train.txt')
            file1 = open(frm_list_fname, 'r')
            while True:
                img_fname = file1.readline().strip()
                if not img_fname:
                    break
                img_file = os.path.join(self.root, 'resized_imgs_train'.format(self.set), img_fname)
                self.files.append((img_file, img_fname))

            frm_list_fname = os.path.join(self.root, 'annotations', 'frame_list_val.txt')
            file1 = open(frm_list_fname, 'r')
            while True:
                img_fname = file1.readline().strip()
                if not img_fname:
                    break
                img_file = os.path.join(self.root, 'resized_imgs_val'.format(self.set), img_fname)
                self.files.append((img_file, img_fname))
        fst = file_range[0]
        fen = file_range[1]
        self.files = self.files[fst:fen]

        # print(len(self.files))

        if self.set == 'train' and self.cfg.DACS_RANDOM_CROP:
            transform_param = {}
            transform_param['crop_h'] = self.cfg.DATASET.RANDOM_CROP_DIM
            transform_param['crop_w'] = self.cfg.DATASET.RANDOM_CROP_DIM
            self.dacs_random_crop = dacs_build_aug(transform_param, is_train=True, dataset='cityscapes')


    def load_img(self, file, size, interpolation, rgb):
        img = Image.open(file)
        if rgb:
            img = img.convert('RGB')
        if size:
            img = img.resize(size, interpolation)
        return np.asarray(img, np.float32)

    def get_image(self, file):
        return self.load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        if self.set == 'train':
            img_file, name = self.files[index]
            image = self.get_image(img_file)
            image = self.preprocess(image).copy()
            semantic = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8) # dummy semantic label
            if self.cfg.DACS_RANDOM_CROP:
                image = image.transpose((1, 2, 0))
                image, semantic = self.dacs_random_crop(image, semantic, train_mode=True)
                image = image.transpose((2, 0, 1))
                # # visual
                # if self.cfg.DEBUG:
                #     self.visualize(name, image, semantic, None, None, None, None, mode='after_crop', train_mode='train_uda')
                return image, semantic, np.array(image.shape), name
        else:
            img_file, name = self.files[index]
            image = self.get_image(img_file)
            image = self.preprocess(image).copy()
            return image, name



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
        # if num_classes == 16:
        return _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID_16
        # elif num_classes == 19:
        #     return _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID_19


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
            colormap[0] =  [128, 64, 128]  # road; 804080ff
            colormap[1] =  [128, 64, 128] # [244, 35, 232]  # sidewalk; f423e8ff # TODO
            colormap[2] =  [70, 70, 70] # building; 464646ff
            colormap[3] =  [102, 102, 156]  # wall; 666699ff
            colormap[4] =  [190, 153, 153]  # fence; be9999ff
            colormap[5] =  [153, 153, 153]  # pole; 999999ff
            colormap[6] =  [250, 170, 30]  # traffic-light; faaa1eff
            colormap[7] =  [220, 220, 0]  # traffic-sign; dcdc00ff
            colormap[8] =  [107, 142, 35]  # vegetation; 6b8e23ff
            colormap[9] =  [70, 130, 180]  # sky; 4682b4ff
            colormap[10] = [220, 20, 60]  # person; dc143cff
            colormap[11] = [220, 20, 60] # [255, 0, 0]  # rider; ff0000ff # TODO
            colormap[12] =  [0, 0, 142]  # car; 00008eff
            colormap[13] =  [0, 60, 100]  # bus; 003c64ff
            colormap[14] =  [0, 0, 230]  # motocycle, 0000e6ff
            colormap[15] =  [119, 11, 32]  # bicycle, 770b20ff

        if num_classes == 2:
            colormap[0] = [0,0,0] # [128, 64, 128]  # road; 804080ff
            colormap[1] = [0,0,0] # [128, 64, 128] # [244, 35, 232]  # sidewalk; f423e8ff # TODO
            colormap[2] = [0,0,0] # [70, 70, 70] # building; 464646ff
            colormap[3] = [0,0,0] # [102, 102, 156]  # wall; 666699ff
            colormap[4] = [0,0,0] # [190, 153, 153]  # fence; be9999ff
            colormap[5] = [0,0,0] # [153, 153, 153]  # pole; 999999ff
            colormap[6] = [0,0,0] # [250, 170, 30]  # traffic-light; faaa1eff
            colormap[7] = [0,0,0] # [220, 220, 0]  # traffic-sign; dcdc00ff
            colormap[8] = [0,0,0] # [107, 142, 35]  # vegetation; 6b8e23ff
            colormap[9] = [0,0,0] # [70, 130, 180]  # sky; 4682b4ff
            colormap[10] = [220, 20, 60]  # person; dc143cff
            colormap[11] = [220, 20, 60] # [255, 0, 0]  # rider; ff0000ff # TODO
            colormap[12] = [0,0,0] # [0, 0, 142]  # car; 00008eff
            colormap[13] = [0,0,0] # [0, 60, 100]  # bus; 003c64ff
            colormap[14] = [0,0,0] # [0, 0, 230]  # motocycle, 0000e6ff
            colormap[15] = [0,0,0] # [119, 11, 32]  # bicycle, 770b20ff

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


    def visualize(self, name, image, semantic, center, center_w, offset, offset_w, visdepth=False, depth=None, mode=None, train_mode=None,  foreground=None, semantic_weights=None):
        save_dir = '/media/suman/DATADISK2/apps/experiments/CVPR2022/cityscapes_panoptic_gt_label_visualization_Jan_15/cityscapes_panop_jan04_2022_v2/{}'.format(train_mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('dir created {}'.format(save_dir))
        fname = name.split('.')[0]
        if 'train_oracle' in train_mode or 'val' in train_mode:
            fname_center = '{}_center_{}'.format(fname, mode)
            fname_center_w = '{}_center_w_{}'.format(fname, mode)
            fname_offset = '{}_offset_{}'.format(fname, mode)
            fname_offset_w = '{}_offset_w_{}'.format(fname, mode)
            fname_semantic = '{}_semantic_{}'.format(fname, mode)
            im4Vis = image.copy()
            im4Vis = im4Vis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
            im4Vis += self.cfg.TRAIN.IMG_MEAN
            im4Vis = im4Vis[:, :, ::-1]  # BGR --> RGB
            offsetVis = offset.copy()
            offsetVis = offsetVis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
            save_heatmap_image(im4Vis, center[0, :], save_dir, fname_center, ratio=0.5, DEBUG=self.cfg.DEBUG)
            save_heatmap_image(im4Vis, center_w, save_dir, fname_center_w, ratio=0.5, DEBUG=self.cfg.DEBUG)
            save_offset_image_v2(im4Vis, offsetVis, save_dir, fname_offset, ratio=0.5, DEBUG=self.cfg.DEBUG)
            save_heatmap_image(im4Vis, offset_w, save_dir, fname_offset_w, ratio=0.5, DEBUG=self.cfg.DEBUG)
            save_annotation(semantic, save_dir, fname_semantic, add_colormap=True, colormap=self.create_label_colormap(16), image=None, blend_ratio=0.7, DEBUG=self.cfg.DEBUG)
            if 'val' in train_mode:
                fname_foreground = '{}_foreground_{}'.format(fname, mode)
                fname_semantic_weights = '{}_semantic_weights_{}'.format(fname, mode)
                save_heatmap_image(im4Vis, foreground, save_dir, fname_foreground, ratio=0.5, DEBUG=self.cfg.DEBUG)
                save_heatmap_image(im4Vis, semantic_weights, save_dir, fname_semantic_weights, ratio=0.5, DEBUG=self.cfg.DEBUG)
        elif 'train_uda' in train_mode:
            fname_semantic = '{}_semantic_{}'.format(fname, mode)
            im4Vis = image.copy()
            im4Vis = im4Vis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
            im4Vis += self.cfg.TRAIN.IMG_MEAN
            im4Vis = im4Vis[:, :, ::-1]  # BGR --> RGB
            save_annotation(semantic, save_dir, fname_semantic, add_colormap=True, colormap=self.create_label_colormap(16), image=im4Vis, blend_ratio=1.0, DEBUG=self.cfg.DEBUG)