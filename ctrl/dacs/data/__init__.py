import json

from ctrl.dacs.data.base import *
from ctrl.dacs.data.cityscapes_loader import cityscapesLoader
from ctrl.dacs.data.gta5_dataset import GTA5DataSet
from ctrl.dacs.data.synthia_dataset import SynthiaDataSet


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSet
    }[name]

# get_data_path(name, cfg.DATA_DIRECTORY_SOURCE, cfg.DATA_DIRECTORY_TARGET)
def get_data_path(name, data_dir_source, data_dir_target):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return data_dir_target
        # return cfg.DATA_DIRECTORY_TARGET
        # return '/media/suman/DATADISK2/apps/datasets/cityscapes_4_panoptic_deeplab/cityscapes'
        # return '../data/CityScapes/'
    if name == 'gta' or name == 'gtaUniform':
        return '/media/suman/CVLHDD/apps/datasets/GTA5/'
        # return '../data/gta/'
    if name == 'synthia':
        return data_dir_source
        # return cfg.DATA_DIRECTORY_SOURCE
        # return '/media/suman/CVLHDD/apps/datasets/Synthia/RAND_CITYSCAPES'
        # return '../data/RAND_CITYSCAPES'
