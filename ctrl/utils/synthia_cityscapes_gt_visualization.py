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
from matplotlib import pyplot as plt
import PIL



#
# def preprocess(self, image):
#     image = image[:, :, ::-1]  # change to BGR
#     image -= self.mean
#     return image.transpose((2, 0, 1))

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

def _colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image


def visualize_depth_gt(dataset='synthia', mode='train', semantic=None, depth=None, fname=None, out_path=None, img=None, img_mean=None):
    str1 = fname.split('.')

    f1 = os.path.join(out_path, '{}.png'.format(str1[0].replace('_gtFine_panoptic', '_pseduo_depth_label')))
    im = PIL.Image.fromarray((_colorize(depth, cmap="plasma") * 255).astype(np.uint8))
    im.save(os.path.join(f1))

    f1 = os.path.join(out_path, '{}.png'.format(str1[0].replace('_gtFine_panoptic', '_semanitc_label')))
    im = Image.fromarray((semantic * 100).astype(np.uint8))
    im = im.convert('RGB')
    im.save(f1)

    # before crop
    # f1 = os.path.join(out_path, '{}.png'.format(str1[0].replace('_gtFine_panoptic', '_input_img')))
    # img = img.transpose((1, 2, 0))
    # img += img_mean
    # img = img[:, :, ::-1]  # change to BGR
    # img = img.astype(np.uint8)
    # im = Image.fromarray(img)
    # im = im.convert('RGB')
    # im.save(f1)

    # after crop

    f1 = os.path.join(out_path, '{}.png'.format(str1[0].replace('_gtFine_panoptic', '_input_img')))
    im = img.transpose((1, 2, 0))
    im += img_mean
    im = im[:, :, ::-1]  # change to BGR
    im = im.astype(np.uint8)
    im = Image.fromarray(im)
    im = im.convert('RGB')
    im.save(f1)



def visualize_panoptic_gt_calss_wise(dataset='synthia', semantic=None, foreground=None, center_cw=None, offset_cw=None, semantic_weights=None, center_w_cw=None, offset_w_cw=None, out_path=None, fname=None):
    new_f = fname.split('.')[0]

    f1 = os.path.join(out_path, '{}_center_cw.png'.format(new_f))
    center_cw_max = np.amax(center_cw, axis=0)
    im = Image.fromarray(center_cw_max.squeeze() * 200)
    im = im.convert('RGB')
    im.save(f1)
    center_w_cw_max = np.amax(center_w_cw, axis=0)
    f1 = os.path.join(out_path, '{}_center_weights_cw.png'.format(new_f))
    im = Image.fromarray(center_w_cw_max * 200)
    im = im.convert('RGB')
    im.save(f1)

    offset_cw = np.sum(offset_cw, axis=0)
    f1 = os.path.join(out_path, '{}_offset_x_cw.png'.format(new_f))
    im = Image.fromarray(offset_cw[0, :, :] * -1)
    im = im.convert('RGB')
    im.save(f1)
    f1 = os.path.join(out_path, '{}_offset_y_cw.png'.format(new_f))
    im = Image.fromarray(offset_cw[1, :, :] * -1)
    im = im.convert('RGB')
    im.save(f1)
    offset_w_cw = np.amax(offset_w_cw, axis=0)
    f1 = os.path.join(out_path, '{}_offset_weights_cw.png'.format(new_f))
    im = Image.fromarray(offset_w_cw * 200)
    im = im.convert('RGB')
    im.save(f1)


def visualize_panoptic_gt(semantic=None,
                            center=None,
                            center_weights=None,
                            offset=None,
                            offset_weights=None,
                            out_path=None,
                            fname=None
                            ):

    str1 = fname.split('.')
    f1 = os.path.join(out_path, '{}_center.png'.format(str1[0]))
    im = Image.fromarray(center.squeeze() * 100)
    im = im.convert('RGB')
    im.save(f1)
    f1 = os.path.join(out_path, '{}_center_weights.png'.format(str1[0]))
    im = Image.fromarray(center_weights * 100)
    im = im.convert('RGB')
    im.save(f1)

    # f1 = os.path.join(out_path, '{}_offset_x.png'.format(str1[0]))
    # im = Image.fromarray(offset[0, :, :] * -1)
    # im = im.convert('RGB')
    # im.save(f1)
    # f1 = os.path.join(out_path, '{}_offset_y.png'.format(str1[0]))
    # im = Image.fromarray(offset[1, :, :] * -1)
    # im = im.convert('RGB')
    # im.save(f1)
    # f1 = os.path.join(out_path, '{}_offset_weights.png'.format(str1[0]))
    # im = Image.fromarray(offset_weights * 100)
    # im = im.convert('RGB')
    # im.save(f1)


def visualize_panoptic_gt_old(dataset='synthia', mode='train', semantic=None, foreground=None, center=None, offset=None,
                          semantic_weights=None, center_weights=None, offset_weights=None, out_path=None, fname=None, depth=None):

    str1=fname.split('.')

    if dataset == 'synthia':
        f1 = os.path.join(out_path, '{}_center.png'.format(str1[0]))
        im = Image.fromarray(center.squeeze() * 100)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_center_weights.png'.format(str1[0]))
        im = Image.fromarray(center_weights * 100)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_offset_x.png'.format(str1[0]))
        im = Image.fromarray(offset[0, :, :] * -1)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_offset_y.png'.format(str1[0]))
        im = Image.fromarray(offset[1, :, :] * -1)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_offset_weights.png'.format(str1[0]))
        im = Image.fromarray(offset_weights * 100)
        im = im.convert('RGB')
        im.save(f1)

    elif dataset == 'cityscapes' and mode == 'train':

        f1 = os.path.join(out_path, '{}_semantic.png'.format(str1[0]))
        im = Image.fromarray((semantic*100).astype(np.uint8))
        im = im.convert('RGB')
        im.save(f1)

        f1 = os.path.join(out_path, '{}_center.png'.format(str1[0]))
        im = Image.fromarray((center.squeeze()*255))
        # im = Image.fromarray((center.squeeze() * 255).astype(np.uint8))
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_center_weights.png'.format(str1[0]))
        mask = center_weights == 1
        center_weights[mask] = 64.0
        im = Image.fromarray(center_weights)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_offset_x.png'.format(str1[0]))
        im = Image.fromarray(offset[0, :, :] * -5)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_offset_y.png'.format(str1[0]))
        im = Image.fromarray(offset[1, :, :] * -5)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_offset_weights.png'.format(str1[0]))
        mask = offset_weights == 1
        offset_weights[mask] = 128.0
        im = Image.fromarray(offset_weights)
        im = im.convert('RGB')
        im.save(f1)

    elif dataset == 'cityscapes' and mode == 'val':
        f1 = os.path.join(out_path, '{}_semantic.png'.format(str1[0]))
        im = Image.fromarray((semantic * 100).astype(np.uint8))
        im = im.convert('RGB')
        im.save(f1)

        f1 = os.path.join(out_path, '{}_center.png'.format(str1[0]))
        im = Image.fromarray((center.squeeze() * 255))
        # im = Image.fromarray((center.squeeze() * 255).astype(np.uint8))
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_center_weights.png'.format(str1[0]))
        mask = center_weights == 1
        center_weights[mask] = 64.0
        im = Image.fromarray(center_weights)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_offset_x.png'.format(str1[0]))
        im = Image.fromarray(offset[0, :, :] * -5)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_offset_y.png'.format(str1[0]))
        im = Image.fromarray(offset[1, :, :] * -5)
        im = im.convert('RGB')
        im.save(f1)
        f1 = os.path.join(out_path, '{}_offset_weights.png'.format(str1[0]))
        mask = offset_weights == 1
        offset_weights[mask] = 128.0
        im = Image.fromarray(offset_weights)
        im = im.convert('RGB')
        im.save(f1)
