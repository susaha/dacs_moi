import numpy as np
from ctrl.utils.serialization import json_load
import os
from ctrl.dataset.gen_panoptic_gt_labels_v2 import PanopticTargetGenerator
import logging
from PIL import Image, ImageOps
from torch.utils import data
import json
from ctrl.dacs_old.data.build_aug import dacs_build_augV5, dacs_build_augV6
from ctrl.utils.panoptic_deeplab.save_annotation import save_heatmap_image, save_offset_image_v2, save_annotation


# as the semantic class id in mapillary panoptic gt labels are same as cityscapres calss id
# we can use these cityscapes class ids for mapillary too
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
_CITYSCAPES_THING_LIST_16 = [10, 11, 12, 13, 14, 15] # continuous trainIds (these ids are same for all datasets)
CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'light', 'sign', 'vegetation', 'sky', 'person', 'rider', 'car', 'bus', 'motocycle', 'bicycle']

class MapillaryDataSet(data.Dataset):

    def __init__(self, root, list_path, set='val', max_iters=None, crop_size=(512, 1024),
                 mean=(128, 128, 128), load_labels=True, info_path=None, labels_size=(1024, 2048),
                 transform=None, joint_transform=None, cfg=None, gen_panop_labels=False, scale_label=True, viz_outpath=None):

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/dataset/mapillary_panop_jan04_2022_v2.py --> class MapillaryDataSet --> __init__() +++')
        self.gen_panop_labels = gen_panop_labels
        self.root = root
        self.cfg = cfg
        self.set = set
        self.image_size = crop_size
        # self.labels_size = labels_size
        self.mean = mean
        self.load_labels = load_labels

        # info_path = '/home/suman/apps/code/CVPR2022/cvpr2022/ctrl/dataset/cityscapes_list/info16class.json' # remove
        # self.info = json_load(info_path)
        # self.class_names = np.array(self.info['label'], dtype=str) # we keep here cityscapes class names only

        self.class_names = np.array(CLASS_NAMES, dtype=str)
        self.scale_label = scale_label
        self.bg_idx = 0 # the panoptic gt labels are color png images, and pixels which has [0,0,0] rgb values have semantic class void
        self.viz_outpath = viz_outpath

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
                                                        small_instance_weight=self.small_instance_weight, dataset='cityscapes') # mapillary has the same id and itrainid encoding as cityscapes, so passing dataset='cityscapes' is ok

        self.files = []
        json_filename = '{}/{}_panoptic.json'.format(cfg.DATA_DIRECTORY_TARGET, self.set)
        dataset = json.load(open(json_filename))
        for ann in dataset['annotations']:
            name = ann['file_name']
            imageId = ann['image_id']
            img_file = '{}/{}_imgs/{}'.format(cfg.DATA_DIRECTORY_TARGET, self.set, '{}.jpg'.format(imageId))
            panop_label_file = '{}/{}_labels/{}'.format(cfg.DATA_DIRECTORY_TARGET, self.set, name)
            segments_info = ann['segments_info']
            self.files.append((img_file, panop_label_file, segments_info, name))

        if self.set == 'train' and self.cfg.DACS_RANDOM_CROP:
            from ctrl.dacs_old.data.build_aug import dacs_build_aug, dacs_build_augV4
            transform_param = {}
            transform_param['crop_h'] = self.cfg.DATASET.RANDOM_CROP_DIM
            transform_param['crop_w'] = self.cfg.DATASET.RANDOM_CROP_DIM
            if self.gen_panop_labels:
                self.dacs_random_crop = dacs_build_augV4(transform_param, is_train=True, dataset='cityscapes')
            else:
                self.dacs_random_crop = dacs_build_aug(transform_param, is_train=True, dataset='cityscapes')


    def __len__(self):
        return len(self.files)

    def load_img(self, file, size, interpolation):
        img = Image.open(file)
        # compute the downsample ratio dsr
        w1 = size[0]  # 1024
        h1 = size[1]  # 768
        w2, h2 = img.size
        # print('__getitem__() : image.size : (w:{}, h:{}) before resize'.format(w2, h2))
        dsr = h1 / h2
        size_new = [int(w2*dsr), int(h2*dsr)]
        img = img.convert('RGB')
        img = img.resize(size_new, interpolation)
        return np.asarray(img, np.float32), size_new

    def get_image(self, file):
        resized_img, size_new = self.load_img(file, self.image_size, Image.BICUBIC)
        return resized_img, size_new

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))


    def __getitem__(self, index):

        img_file, panop_label_file, segment_info, name = self.files[index]
        image, size_new = self.get_image(img_file)
        image = self.preprocess(image).copy()
        # print('__getitem__() : image.shape: {} after resize'.format(image.shape))

        if self.set == 'train':

            if self.gen_panop_labels:  # this is reruired for Oracle training or training on UDA setting: cityscapes --> mapillary
                # get the panoptic gt labels
                label_panop = Image.open(panop_label_file)
                label_panop = label_panop.resize(size_new, Image.NEAREST)
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
                    # print('__getitem__() : image.shape: {} after crop'.format(image.shape))
                    # print('---')

                # # visual
                # if self.cfg.DEBUG:
                #     self.visualize(name, image, semantic, None, None, None, None, mode='after_crop', train_mode='train_uda')

                return image, semantic, np.array(image.shape), name

        else:
            # this part is executed during evaluation, so we collect the panoptic gt labels in a dictionary "label_panop_dict"
            # and pass it to the panoptic-deeplabe panoptic evaluation script

            # semseg_label = self.get_labels(label_file)
            # semseg_label = self.map_labels(semseg_label).copy()
            label_panop = Image.open(panop_label_file)
            # print('__getitem__() : label_panop.size: {} before resize'.format(label_panop.size))
            label_panop = label_panop.resize(size_new, Image.NEAREST)
            label_panop = np.asarray(label_panop, dtype=np.float32)  # the id values are > 255, we need np.float32
            # print('__getitem__() : label_panop.shape: {} after resize'.format(label_panop.shape))
            label_panop_dict = self.target_transform(label_panop, segment_info, mode='val')
            # print('__getitem__() : label_panop_dict[semantic].shape: {} after transform'.format(label_panop_dict['semantic'].shape))
            # label_panop_dict['semantic'] = torch.as_tensor(semseg_label.astype('long'))  # semanitc gt label

            # for visualization only
            # if self.cfg.DEBUG:
            #     semantic = label_panop_dict['semantic']
            #     center = label_panop_dict['center']
            #     center_w = label_panop_dict['center_weights']
            #     offset = label_panop_dict['offset']
            #     offset_w = label_panop_dict['offset_weights']
            #     foreground = label_panop_dict['foreground']
            #     semantic_weights = label_panop_dict['semantic_weights']
            #     self.visualize(name, image, semantic, center, center_w, offset, offset_w, mode='before_crop',
            #                    train_mode='val', foreground=foreground, semantic_weights=semantic_weights)

            return image.copy(), label_panop_dict, np.array(image.shape), name

    @staticmethod
    def train_id_to_eval_id(num_classes):
        return _CITYSCAPES_PANOPTIC_TRAIN_ID_TO_EVAL_ID_16

    @staticmethod
    def rgb2id(color):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    @staticmethod
    def create_label_colormap(num_classes):
        colormap = np.zeros((256, 3), dtype=np.uint8)
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

