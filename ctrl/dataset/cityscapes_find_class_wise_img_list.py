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
from ctrl.utils.panoptic_deeplab.save_annotation import save_heatmap_image, save_offset_image_v2, save_annotation, save_instance_annotation
from visualize_during_main_training import get_instance_visual

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

# _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID_19 = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 0] # ids
# _CITYSCAPES_THING_LIST_19 = [11, 12, 13, 14, 15, 16, 17, 18]  # continuous trainIds

class CityscapesDataSet(data.Dataset):
    '''
    crop_size: is the image size (512, 1024)
    labels_size: is the GT label size, at training time you sue (512, 1024) at test time (1024, 2048)
    to make dataloading faster during UDA training, we can skip loading the panoptic labels by setting self.gen_panop_labels=False
    but for oracel training you need to GT label
    Note: self.gen_panop_labels is used during training, for evaluation the dataloader reutrn panoptic gt labels required for panoptic evaluation
    '''
    def __init__(self, root, list_path, set='val', max_iters=None, crop_size=(512, 1024),
                 mean=(128, 128, 128), load_labels=True, info_path=None, labels_size=(1024, 2048),
                 transform=None, joint_transform=None, cfg=None, gen_panop_labels=False, cid=None):

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/dataset/cityscapes_panop_nov12_2021.py --> class CityscapesDataSet --> __init__() +++')
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
        # self.mapping = np.array(self.info['label2train'], dtype=int)
        # self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        # for source_label, target_label in self.mapping:
        #     self.map_vector[source_label] = target_label

        cfgp = cfg['PANOPTIC_TARGET_GENERATOR']
        self.ignore_label = cfgp['IGNORE_LABEL']
        self.label_divisor = cfgp['LABEL_DIVISOR']
        self.cityscapes_thing_list = _CITYSCAPES_THING_LIST_16
        print('*** self.cityscapes_thing_list: {} ***'.format(self.cityscapes_thing_list))
        self.ignore_stuff_in_offset = cfgp['IGNORE_STUFF_IN_OFFSET']
        self.small_instance_area = cfgp['SMALL_INSTANCE_AREA']
        self.small_instance_weight = cfgp['SMALL_INSTANCE_WEIGHT']
        self.sigma = cfgp['SIGMA']
        self.target_transform = PanopticTargetGenerator(self.logger, self.ignore_label, self.rgb2id, self.cityscapes_thing_list,
                                                        sigma=8, ignore_stuff_in_offset=self.ignore_stuff_in_offset,
                                                        small_instance_area=self.small_instance_area,
                                                        small_instance_weight=self.small_instance_weight,  dataset='cityscapes')

        # self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)
        self.files = []
        if self.set == 'train':
            json_filename = os.path.join(self.root, 'gtFineV2', 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}_trainId.json'.format(self.set))
        elif self.set == 'val':
            json_filename = os.path.join(self.root, 'gtFineV2', 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}.json'.format(self.set))

        dataset = json.load(open(json_filename))
        for ann in dataset['annotations']:
            name = ann['file_name']
            img_file = os.path.join(self.root, 'leftImg8bit', self.set, name.split('_')[0], name.replace('_gtFine_panoptic', '_leftImg8bit'))
            label_file = os.path.join(self.root, 'gtFineV2', self.set, name.split('_')[0], name.replace('_gtFine_panoptic', '_gtFine_labelIds'))
            if self.set == 'train':
                panop_label_file = os.path.join(self.root, 'gtFineV2', 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}_trainId'.format(self.set), name)
            elif self.set == 'val':
                panop_label_file = os.path.join(self.root, 'gtFineV2', 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}'.format(self.set), name)

            segments_info = ann['segments_info']
            self.files.append((img_file, label_file, panop_label_file, segments_info, name))

        if self.set == 'train' and self.cfg.DACS_RANDOM_CROP:
            transform_param = {}
            transform_param['crop_h'] = self.cfg.DATASET.RANDOM_CROP_DIM
            transform_param['crop_w'] = self.cfg.DATASET.RANDOM_CROP_DIM
            if self.gen_panop_labels:
                self.dacs_random_crop = dacs_build_augV4(transform_param, is_train=True, dataset='cityscapes')
                # self.dacs_random_crop = dacs_build_augV2(transform_param, is_train=True, dataset='cityscapes')
            else:
                self.dacs_random_crop = dacs_build_aug(transform_param, is_train=True, dataset='cityscapes')

        self.fdist_classes = [6,7,10,11,12,13,14,15]

        self.cid = cid
        self.text_file = open('cityscapes_valset_class_wise_img_list_cid_{}.txt'.format(cid), 'w')
        self.write_file('class_train_id # img_file_name # num_pixels')



    # def map_labels(self, input_):
    #     return self.map_vector[input_.astype(np.int64, copy=False)]

    def write_file(self, line):
        self.text_file.write('{}\n'.format(line))  # python will convert \n to os.linesep

    def close_file(self):
        self.text_file.close()  # you can omit in most cases as the destructor will call it
        print('file closed')

    def load_img(self, file, size, interpolation, rgb):
        img = Image.open(file)
        if rgb:
            img = img.convert('RGB')
        if size:
            img = img.resize(size, interpolation)
        return np.asarray(img, np.float32)

    def get_image(self, file):
        return self.load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    # def get_labels(self, file):
    #     return self.load_img(file, self.labels_size, Image.NEAREST, rgb=False)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):

        if self.set == 'train':
            img_file, label_file, panop_label_file, segment_info, name = self.files[index]
            image = self.get_image(img_file)
            image = self.preprocess(image).copy()

            if self.gen_panop_labels:  # this is reruired for Oracle training or training on UDA setting: cityscapes --> mapillary
                # get the panoptic gt labels
                label_panop = Image.open(panop_label_file)
                label_panop = label_panop.resize(self.labels_size, Image.NEAREST)
                label_panop = np.asarray(label_panop, dtype=np.float32)  # the id values are > 255, we need np.float32

                #  apply random crop on image and label_panop
                if self.cfg.DACS_RANDOM_CROP:
                    image = image.transpose((1, 2, 0))
                    image, label_panop, depth = self.dacs_random_crop(image, label_panop, depth=None, use_depth=False, train_mode=True)
                    image = image.transpose((2, 0, 1))

                # generate panoptic gt labels
                label_panop_dict = self.target_transform(label_panop, segment_info, mode='train')
                semantic = label_panop_dict['semantic']
                center = label_panop_dict['center']
                center_w = label_panop_dict['center_weights']
                offset = label_panop_dict['offset']
                offset_w = label_panop_dict['offset_weights']

                # visual
                # if self.cfg.DEBUG:
                #     self.visualize(name, image, semantic, center, center_w, offset, offset_w, mode='after_crop', train_mode='train_oracle')

                # if self.cfg.DACS_RANDOM_CROP:
                #     image = image.transpose((1, 2, 0))
                #     center = center.transpose((1, 2, 0))
                #     offset = offset.transpose((1, 2, 0))
                #     image, semantic, center, center_w, offset, offset_w, _ = \
                #         self.dacs_random_crop(image, semantic, center, center_w, offset, offset_w, use_depth=False, train_mode=True)
                #     image = image.transpose((2, 0, 1))
                #     center = center.transpose((2, 0, 1))
                #     offset = offset.transpose((2, 0, 1))

                return image, semantic.copy(), center.copy(), center_w.copy(), offset.copy(), offset_w.copy(), np.array(image.shape), name
            else:
                # this part is reuqired for UDA training - we dont need semanitc label for UDA setting
                # semantic = self.get_labels(label_file)
                # semantic = self.map_labels(semantic).copy()
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

            img_file, label_file, panop_label_file, segment_info, name = self.files[index]
            # image = self.get_image(img_file)
            # image = self.preprocess(image).copy()
            label_panop = Image.open(panop_label_file)
            label_panop = label_panop.resize(self.labels_size, Image.NEAREST)
            label_panop = np.asarray(label_panop, dtype=np.float32)  # the id values are > 255, we need np.float32
            label_panop_dict = self.target_transform(label_panop, segment_info, mode='val')

            semantic = label_panop_dict['semantic']
            ucids = np.unique(semantic)
            if self.cid in ucids:
                mask = semantic == self.cid
                self.write_file('{} # {}# {}'.format(self.cid, name, mask.sum()))

            # return image.copy(), label_panop_dict, np.array(image.shape), name
            return name


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


    def visualize(self, name, image, semantic, center, center_w, offset, offset_w, visdepth=False, depth=None, mode=None, train_mode=None,  foreground=None, semantic_weights=None, instance=None):
        fname = name.split('.')[0]
        str1 = fname.split('_')
        folder_name = str1[0] + '_' + str1[1] + '_' + str1[2]
        save_dir = '/media/suman/DATADISK2/apps/experiments/CVPR2022/cityscapes_panoptic_gt_label_visualization_march_01_2022/{}/{}'.format(train_mode, folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('dir created {}'.format(save_dir))

        if 'train_oracle' in train_mode or 'val' in train_mode:
            fname_center = 'center' # '{}_center_{}'.format(fname, mode)
            fname_center_w = 'center_w' # '{}_center_w_{}'.format(fname, mode)
            fname_offset = 'offset' # '{}_offset_{}'.format(fname, mode)
            fname_offset_w = 'offset_w' # '{}_offset_w_{}'.format(fname, mode)
            fname_semantic = 'semantic' # '{}_semantic_{}'.format(fname, mode)
            fname_instance = 'instance' # '{}_instance_{}'.format(fname, mode)
            im4Vis = image.copy()
            im4Vis = im4Vis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
            im4Vis += self.cfg.TRAIN.IMG_MEAN
            im4Vis = im4Vis[:, :, ::-1]  # BGR --> RGB
            offsetVis = offset.copy()
            offsetVis = offsetVis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
            save_heatmap_image(im4Vis, center[0, :], save_dir, fname_center, ratio=0.7, DEBUG=self.cfg.DEBUG)
            save_heatmap_image(im4Vis, center_w, save_dir, fname_center_w, ratio=0.5, DEBUG=self.cfg.DEBUG)
            save_offset_image_v2(im4Vis, offsetVis, save_dir, fname_offset, ratio=0.7, DEBUG=self.cfg.DEBUG)
            save_heatmap_image(im4Vis, offset_w, save_dir, fname_offset_w, ratio=0.5, DEBUG=self.cfg.DEBUG)
            save_annotation(semantic, save_dir, fname_semantic, add_colormap=True,colormap=self.create_label_colormap(16),
                            image=im4Vis, blend_ratio=0.7, DEBUG=self.cfg.DEBUG)
            if 'val' in train_mode:
                fname_foreground = 'foreground' # '{}_foreground_{}'.format(fname, mode)
                fname_semantic_weights = 'semantic_weights' # '{}_semantic_weights_{}'.format(fname, mode)
                save_heatmap_image(im4Vis, foreground, save_dir, fname_foreground, ratio=0.5, DEBUG=self.cfg.DEBUG)
                save_heatmap_image(im4Vis, semantic_weights, save_dir, fname_semantic_weights, ratio=0.5, DEBUG=self.cfg.DEBUG)

            # label_instance_source_vis = get_instance_visual(label_instance_source[bid].detach().cpu().numpy().copy().astype(int), )
            save_instance_annotation(instance, save_dir, fname_instance, image=im4Vis, blend_ratio=0.7,  DEBUG=self.cfg.DEBUG)

        elif 'train_uda' in train_mode:
            fname_semantic = 'semantic' # '{}_semantic_{}'.format(fname, mode)
            im4Vis = image.copy()
            im4Vis = im4Vis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
            im4Vis += self.cfg.TRAIN.IMG_MEAN
            im4Vis = im4Vis[:, :, ::-1]  # BGR --> RGB
            save_annotation(semantic, save_dir, fname_semantic, add_colormap=True, colormap=self.create_label_colormap(16), image=im4Vis, blend_ratio=1.0, DEBUG=self.cfg.DEBUG)