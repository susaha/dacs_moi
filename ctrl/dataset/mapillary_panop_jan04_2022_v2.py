import numpy as np
from ctrl.utils.serialization import json_load
import os
from ctrl.dataset.gen_panoptic_gt_labels_v2 import PanopticTargetGenerator
import logging
from PIL import Image, ImageOps
from torch.utils import data
import json
from ctrl.dacs_old.data.build_aug import dacs_build_augV5, dacs_build_augV6
from ctrl.utils.panoptic_deeplab.save_annotation import save_heatmap_image, save_offset_image_v2, save_annotation, save_instance_annotation


def resize_with_pad(target_size, image, resize_type, fill_value=0, is_label=False):
    if target_size is None:
        if is_label:
            return np.array(image, dtype=np.uint8)
        else:
            return np.array(image)

    # find which size to fit to the target size
    target_ratio = target_size[0] / target_size[1]
    image_ratio = image.size[0] / image.size[1]

    if image_ratio > target_ratio:
        resize_ratio = target_size[0] / image.size[0]  # target_widht / image_widht
        new_image_shape = (target_size[0], int(image.size[1] * resize_ratio))
    else:
        resize_ratio = target_size[1] / image.size[1]  # target_height / image_height
        new_image_shape = (int(image.size[0] * resize_ratio), target_size[1])

    image_resized = image.resize(new_image_shape, resize_type)

    if is_label:
        image_resized = np.array(image_resized, dtype=np.uint8)
    else:
        image_resized = np.array(image_resized)

    if image_resized.ndim == 2:
        image_resized = image_resized[:, :, None]

    result = np.ones(target_size[::-1] + [image_resized.shape[2], ], np.float32) * fill_value
    assert image_resized.shape[0] <= result.shape[0]
    assert image_resized.shape[1] <= result.shape[1]
    placeholder = result[:image_resized.shape[0], :image_resized.shape[1]]
    placeholder[:] = image_resized
    return result, new_image_shape


def pad_with_fixed_AS(target_ratio, image, fill_value=0, is_label=False):
    dimW = float(image.size[0])
    dimH = float(image.size[1])
    image_ratio = dimW / dimH
    if target_ratio > image_ratio:
        dimW = target_ratio * dimH
    elif target_ratio < image_ratio:
        dimH = dimW / target_ratio
    else:
        if is_label:
            return np.array(image, dtype=np.uint8)
        else:
            return np.array(image)

    if is_label:
        image = np.array(image, dtype=np.uint8)
    else:
        image = np.array(image)
    result = np.ones((int(dimH), int(dimW), int(image.shape[2])), np.float32) * fill_value
    placeholder = result[:image.shape[0], :image.shape[1]]
    placeholder[:] = image
    return result


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
            transform_param = {}
            transform_param['crop_h'] = self.cfg.DATASET.RANDOM_CROP_DIM
            transform_param['crop_w'] = self.cfg.DATASET.RANDOM_CROP_DIM
            if self.gen_panop_labels:
                self.dacs_random_crop = dacs_build_augV5(transform_param, is_train=True, dataset='mapillary')
            else:
                self.dacs_random_crop = dacs_build_augV6(transform_param, is_train=True, dataset='mapillary')

    def __len__(self):
        return len(self.files)

    def get_image(self, file):
        return Image.open(file).convert('RGB')

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))


    def __getitem__(self, index):

        # image = pad_with_fixed_AS(self.image_size[0] / self.image_size[1], image, fill_value=0, is_label=False)

        img_file, panop_label_file, segment_info, name = self.files[index]
        image = self.get_image(img_file)
        image, new_image_shape = resize_with_pad(self.image_size, image, Image.BICUBIC)
        image = self.preprocess(image)

        if self.set == 'train':
            if self.gen_panop_labels:  # this is reruired for Oracle training or training on UDA setting: mapillary --> cityscapes
                # get the panoptic gt labels
                label_panop = Image.open(panop_label_file)
                label_panop, new_label_shape = resize_with_pad(self.image_size, label_panop, Image.NEAREST, fill_value=self.bg_idx, is_label=True)
                assert new_image_shape == new_label_shape, 'error in __getitem__ of calss MapillaryDataSet --> ctrl/dataset/mapillary_panop_jan04_2022_v2.py'

                # visual
                if self.viz_outpath:
                    train_mode = 'train_with_panoptic'
                    self.visualize_before_crop(name, image, label_panop, train_mode)

                #  apply random crop on image and label_panop
                if self.cfg.DACS_RANDOM_CROP:
                    image = image.transpose((1, 2, 0))
                    # we pass new_image_shape, so that the random cropping is done within the image region and not in the padded region
                    image, label_panop, _ = self.dacs_random_crop(image, label_panop, depth=None, use_depth=False, train_mode=True, new_image_shape=new_image_shape)
                    image = image.transpose((2, 0, 1))

                # generate panoptic gt labels
                label_panop_dict = self.target_transform(label_panop, segment_info, mode='train')
                semantic = label_panop_dict['semantic']
                center = label_panop_dict['center']
                center_w = label_panop_dict['center_weights']
                offset = label_panop_dict['offset']
                offset_w = label_panop_dict['offset_weights']

                # visual
                if self.viz_outpath:
                    train_mode = 'train_with_panoptic'
                    self.visualize_after_crop(name, image, semantic, center, center_w, offset, offset_w, train_mode)

                return image.copy(), semantic.copy(), center.copy(), center_w.copy(), offset.copy(), offset_w.copy(), np.array(image.shape), name

            else:
                semantic = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)  # dummy semantic label
                if self.cfg.DACS_RANDOM_CROP:
                    image = image.transpose((1, 2, 0))
                    # if image.shape[0] != new_image_shape[1] or image.shape[1] != new_image_shape[0]:
                    #     print()
                    image, semantic = self.dacs_random_crop(image, semantic, train_mode=True, new_image_shape=new_image_shape)
                    image = image.transpose((2, 0, 1))

                # visual
                if self.viz_outpath:
                    train_mode = 'train_without_panoptic'
                    self.visualize_after_crop_v2(name, image, train_mode)

                return image.copy(), semantic, np.array(image.shape), name

        else:
            label_panop = Image.open(panop_label_file)
            # note that pad_with_fixed_AS() don't downsample the label_panop
            # label_panop has the same spatial dimension as the mapillary original images, that is they are really large
            # the function pad_with_fixed_AS() takes the aspect ration (i.e. width/hegit = self.image_size[0] / self.image_size[1] = 1024/768 = 1.33)
            # of the input image (defined in cfg.TEST.INPUT_SIZE_TARGET or self.crop_size in this script) as input
            # and use that to create a new dimension for the panoptic label so that we can keep the original shape of the panoptic label
            # and at the same time we can match the aspect ratio of the input image and the gt label
            # e.g. the original mapillary image is resized to 1024 x 768 shape using resize_with_pad() function
            # (the aspect ratio of the original image is maintained and the extra regions are padded)
            # then it is passed to the model and predictions are obtained
            # in the prediction, the aspect ratio is mainted with padding
            # in the gt panoptic label the aspect ratio is maintained with padding
            # so now if you upsample the prediction to the gt label size, you can match pixel to pixel labeling of the prediction and the gt
            # exactly this is done in the evaluation

            # this trows gpu out of memory error as some images are very large 4000 x 6000
            # and the postprocessing step of panoptic deeplab takes lot of gpu memory
            # when computing the center for large images, so for the time being I comment this out
            # label_panop = pad_with_fixed_AS(self.image_size[0] / self.image_size[1], label_panop, fill_value=self.bg_idx, is_label=True) # TODO: DADA ORIGINAL

            # so at the time of evaluation, I am resizing the panoptic labels to shape 768 x 1024 (padding is added wherever required to match the aspect ratio)
            label_panop, _ = resize_with_pad(self.image_size, label_panop, Image.NEAREST, fill_value=self.bg_idx, is_label=True)

            label_panop_dict = self.target_transform(label_panop, segment_info, mode='val')

            # visual
            if self.viz_outpath:
                # I need to apply pad_with_fixed_AS() on image (see below)
                # this is required for visualization (but not for actual training), as the image and the label are in two different shapes
                # the image is in [1024 x 768] and the label has widht and height closer to the original width and height
                # image = self.get_image(img_file)
                # image = pad_with_fixed_AS(self.image_size[0] / self.image_size[1], image, fill_value=0, is_label=False)
                # image, new_image_shape = resize_with_pad(self.image_size, image, Image.BICUBIC)
                # image = np.array(image, dtype=np.float32)
                # image = self.preprocess(image)

                train_mode = 'val_with_panoptic'
                semantic = label_panop_dict['semantic']
                center = label_panop_dict['center']
                center_w = label_panop_dict['center_weights']
                offset = label_panop_dict['offset']
                offset_w = label_panop_dict['offset_weights']
                foreground = label_panop_dict['foreground']
                semantic_weights = label_panop_dict['semantic_weights']
                instance = label_panop_dict['instance']
                self.visualize_all_during_val(name, image, semantic, center, center_w, offset, offset_w, train_mode, foreground, semantic_weights, instance)

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

    def visualize_after_crop_v2(self, name, image, train_mode):
        save_dir = '{}/{}'.format(self.viz_outpath, train_mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('dir created {}'.format(save_dir))
        fname = name.split('.')[0]
        mode = 'before_crop'
        fname_img = '{}_img_{}'.format(fname, mode)
        im4Vis = image.copy()
        im4Vis = im4Vis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
        im4Vis += self.cfg.TRAIN.IMG_MEAN
        im4Vis = im4Vis[:, :, ::-1]  # BGR --> RGB
        pil_image = Image.fromarray(im4Vis.astype(dtype=np.uint8))
        out_fname = '%s/%s.png' % (save_dir, fname_img)
        with open(out_fname, mode='wb') as f:
            pil_image.save(f, 'PNG')

    def visualize_before_crop(self, name, image, panoptic, train_mode):
        save_dir = '{}/{}'.format(self.viz_outpath, train_mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('dir created {}'.format(save_dir))
        fname = name.split('.')[0]
        mode = 'before_crop'
        fname_img = '{}_img_{}'.format(fname, mode)
        im4Vis = image.copy()
        im4Vis = im4Vis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
        im4Vis += self.cfg.TRAIN.IMG_MEAN
        im4Vis = im4Vis[:, :, ::-1]  # BGR --> RGB
        pil_image = Image.fromarray(im4Vis.astype(dtype=np.uint8))
        out_fname = '%s/%s.png' % (save_dir, fname_img)
        with open(out_fname, mode='wb') as f:
            pil_image.save(f, 'PNG')
        fname_panop = '{}_panoptic_{}'.format(fname, mode)
        panop_vis = panoptic.copy()
        pil_image = Image.fromarray(panop_vis.astype(dtype=np.uint8))
        out_fname = '%s/%s.png' % (save_dir, fname_panop)
        with open(out_fname, mode='wb') as f:
            pil_image.save(f, 'PNG')

    def visualize_after_crop(self, name, image, semantic, center, center_w, offset, offset_w, train_mode):
        save_dir = '{}/{}'.format(self.viz_outpath, train_mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('dir created {}'.format(save_dir))
        fname = name.split('.')[0]
        mode = 'after_crop'
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

    def visualize_all_during_val(self, name, image, semantic, center, center_w, offset, offset_w, train_mode, foreground, semantic_weights, instance):
        fname = name.split('.')[0]
        save_dir = '{}/val/{}'.format(self.viz_outpath, fname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('dir created {}'.format(save_dir))

        fname_center = 'center'  # '{}_center_{}'.format(fname, mode)
        fname_center_w = 'center_w'  # '{}_center_w_{}'.format(fname, mode)
        fname_offset = 'offset'  # '{}_offset_{}'.format(fname, mode)
        fname_offset_w = 'offset_w'  # '{}_offset_w_{}'.format(fname, mode)
        fname_semantic = 'semantic'  # '{}_semantic_{}'.format(fname, mode)
        fname_instance = 'instance'  # '{}_instance_{}'.format(fname, mode)
        fname_foreground =  'foreground' # '{}_foreground_{}'.format(fname, mode)
        fname_semantic_weights =  'semantic_weights' # '{}_semantic_weights_{}'.format(fname, mode)
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
        save_annotation(semantic, save_dir, fname_semantic, add_colormap=True, colormap=self.create_label_colormap(16), image=None, blend_ratio=0.7, DEBUG=self.cfg.DEBUG)

        save_heatmap_image(im4Vis, foreground, save_dir, fname_foreground, ratio=0.5, DEBUG=self.cfg.DEBUG)
        save_heatmap_image(im4Vis, semantic_weights, save_dir, fname_semantic_weights, ratio=0.5, DEBUG=self.cfg.DEBUG)
        save_instance_annotation(instance, save_dir, fname_instance, image=im4Vis, blend_ratio=0.7, DEBUG=self.cfg.DEBUG)

