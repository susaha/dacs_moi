import numpy as np
from ctrl.dataset.base_dataset import BaseDataset
from ctrl.dataset.depth import get_depth_dada, get_depth_gasda
import os
from ctrl.dataset.gen_panoptic_gt_labels import PanopticTargetGenerator
from PIL import Image
import numpy as np
import torch
import logging
from PIL import Image, ImageOps
from ctrl.transforms_panop import build_transforms
from ctrl.dacs.data.augmentations import *

'''
SYNTHIA-RAND-CITYSCAPES - CLASS TO ID MAPPING
This below table is copied from the README.txt of SYNTHIA-RAND-CITYSCAPES dataset
Class		    R	G	B	    ID
void		    0	0	0	    0
sky		        70	130	180	    1
Building	    70	70	70	    2
Road		    128	64	128	    3
Sidewalk	    244	35	232	    4
Fence		    64	64	128	    5
Vegetation	    107	142	35	    6
Pole		    153	153	153	    7
Car		        0	0	142	    8
Traffic sign	220	220	0	    9
Pedestrian	    220	20	60	    10
Bicycle		    119	11	32	    11
Motorcycle	    0	0	230	    12
Parking-slot	250	170	160	    13
Road-work	    128	64	64	    14
Traffic light	250	170	30	    15
Terrain		    152	251	152	    16
Rider		    255	0	0	    17
Truck		    0	0	70	    18
Bus		        0	60	100	    19
Train		    0	80	100	    20
Wall		    102	102	156 	21
Lanemarking	    102	102	156	    22
'''

_SYNTHIA_THING_LIST = [10, 11, 12, 13, 14, 15] # [person, rider, car, bus, motorcycle, bicycle]

class SYNTHIADataSetDepth(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        set="all",
        num_classes=16,
        max_iters=None,
        crop_size=(321, 321),
        mean=(128, 128, 128),
        use_depth=False,
        depth_processing='GASDA',
        cfg=None,
        joint_transform=None,
    ):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean, joint_transform, cfg)

        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/dataset/synthia_panop.py -->  class SYNTHIADataSetDepth() : __init__()')

        if num_classes == 16:
            self.id_to_trainid = {
                3: 0,  # road
                4: 1,  # Sidewalk
                2: 2,  # Building
                21: 3,  # Wall
                5: 4,  # Fence
                7: 5,  # Pole
                15: 6,  # Traffic light
                9: 7,  # Traffic sign
                6: 8,  # Vegetation
                1: 9,  # sky
                10: 10,  # Pedestrian or person
                17: 11,  # Rider
                8: 12,  # Car
                19: 13,  # Bus
                12: 14,  # Motorcycle
                11: 15,  # Bicycle
            }
        elif num_classes == 7:
            self.id_to_trainid = {
                # CLASS NAME    : GROUP NAME IN 7 CLASS SETTING
                1: 4,  # sky           : SKY
                2: 1,  # Building      : CONSTRUCTION
                3: 0,  # Road          : FLAT
                4: 0,  # Sidewalk      : FLAT
                5: 1,  # Fence         : CONSTRUCTION
                6: 3,  # Vegetation    : NATURE
                7: 2,  # Pole          : OBJECT
                8: 6,  # Car           : VEHICLE
                9: 2,  # Traffic sign  : OBJECT
                10: 5,  # Pedestrian    : HUMAN
                11: 6,  # Bicycle       : VEHICLE
                15: 2,  # Traffic light : OBJECT
                22: 0   # Lanemarking   : FLAT
            }
        else:
            raise NotImplementedError(f"Not yet supported {num_classes} classes")

        self.cfg = cfg
        self.joint_transform = joint_transform
        self.use_depth = use_depth
        self.depth_processing = depth_processing
        if self.use_depth:
            for (i, file) in enumerate(self.files):
                if self.dataloading_mode == 'panoptic':
                    img_file, label_file, panop_label_file, segment_info, name = file
                    depth_file = self.root / "Depth" / name
                    self.files[i] = (img_file, label_file, panop_label_file, segment_info, depth_file, name)
                else:
                    img_file, label_file, name = file
                    depth_file = self.root / "Depth" / name
                    self.files[i] = (img_file, label_file, depth_file, name)
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"

        if self.dataloading_mode == 'panoptic':
            cfgp = cfg['PANOPTIC_TARGET_GENERATOR']
            self.ignore_label = cfgp['IGNORE_LABEL']
            self.synthia_thing_list = _SYNTHIA_THING_LIST
            self.ignore_stuff_in_offset = cfgp['IGNORE_STUFF_IN_OFFSET']
            self.small_instance_area = cfgp['SMALL_INSTANCE_AREA']
            self.small_instance_weight = cfgp['SMALL_INSTANCE_WEIGHT']
            self.sigma = cfgp['SIGMA']
            self.target_transform = PanopticTargetGenerator(self.logger, self.ignore_label, self.rgb2id, self.synthia_thing_list,
                                                            sigma=8, ignore_stuff_in_offset=self.ignore_stuff_in_offset,
                                                            small_instance_area=self.small_instance_area,
                                                            small_instance_weight=self.small_instance_weight)



        self.img_size = (1280, 760)
        self.input_size = (self.cfg.DATASET.RANDOM_CROP_DIM, self.cfg.DATASET.RANDOM_CROP_DIM)
        self.augmentations = Compose([RandomCrop_gta(self.input_size)])


    def get_metadata(self, name, mode=None):
        label_file = self.root / "parsed_LABELS" / name
        if mode == 'original_only':
            img_file = self.root / "RGB" / name
            return img_file, label_file
        else:
            self.logger.info('ctrl/dataset/synthia_panop.py -->  class SYNTHIADataSetDepth() :get_metadata() --> set proper value for cfg.SYNTHIA_DATALOADING_MODE')
            raise NotImplementedError



    def __getitem__(self, index):

        if self.dataloading_mode == 'panoptic':
            img_file, label_file, panop_label_file, segment_info, depth_file, name = self.files[index]
        else:
            if self.use_depth:
                img_file, label_file, depth_file, name = self.files[index]
            else:
                img_file, label_file, name = self.files[index]

        # read image
        image = Image.open(img_file).convert('RGB')
        label = Image.open(label_file)
        # resize
        image = image.resize(self.img_size, Image.BICUBIC)
        label = label.resize(self.img_size, Image.NEAREST)
        # convert to numpy array
        image = np.asarray(image, np.uint8)
        label = np.asarray(label, np.uint8)
        # apply random crop
        if self.augmentations is not None:
            image, label = self.augmentations(image, label)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


    def get_depth(self, file):
        if self.depth_processing == 'GASDA':
            return get_depth_gasda(self, file, phase='train')
        elif self.depth_processing == 'DADA':
            return get_depth_dada(self, file, phase='train')

    # this function is copied from: /home/suman/apps/code/CVPR2022/panoptic-deeplab/segmentation/data/datasets/cityscapes_panoptic.py
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
        if num_classes == 16:
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
        else:
            raise NotImplementedError('ctrl/dataset/synthia_panop.py -->  def create_label_colormap(num_classes) '
                                      '--> No implementation foudn for num_classes {}!!'.format(num_classes))

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
