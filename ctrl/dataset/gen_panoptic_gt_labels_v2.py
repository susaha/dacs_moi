# ------------------------------------------------------------------------------
# Generates targets for Panoptic-DeepLab.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import numpy as np
import torch
from synthiascripts.cityscapes_labels_16cls import id2label, labels

'''
NOTE-1:
    # if the dataset is cityscapes and mode is validation
    # then we need to convert the seg["category_id"] to trainId
    # in the validation json file (i.e. cityscapes_panoptic_synthia_to_cityscapes_16cls_val.json), the category_id and id are same
    # this is done because the cityscapes evaluation script requires the semanitc ids as ids and not trainIds
    # so we generate the json file for validation where we store category_id and id as ids and not trainids
    # using this script: cityscapesscripts/preparation/createPanopticImgs.py
    # However, this script is for training, for generating the semantic gt labels, we need cat_id to trainIds
    # so I am converting the cat_id to trainIds.
    
    # this modification is not required in panoptic deeplab original code because, 
    it uses two different validation json files in two different places.
    in the dataloader, it uses the validation json file which has trainId 
    and in the panoptic evaluation script, it uses the val json which uses id
    The way I ran experiments with the original panoptic deeplab code is that
    I generated the val json two times, once with trainIds and another with ids
    the json with trainId is used in the cityscapes dataloader script
    and the json with id is used in the panoptic evaluation script
'''
class PanopticTargetGenerator(object):
    """
    Generates panoptic training target for Panoptic-DeepLab.
    Annotation is assumed to have Cityscapes format.
    Arguments:
        ignore_label: Integer, the ignore label for semantic segmentation.
        rgb2id: Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
        thing_list: List, a list of thing classes
        sigma: the sigma for Gaussian kernel.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
        ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in semantic segmentation branch,
            crowd region is ignored in the original TensorFlow implementation.
    """
    def __init__(self, logger, ignore_label, rgb2id, thing_list, sigma=8, ignore_stuff_in_offset=False,
                 small_instance_area=0, small_instance_weight=1, ignore_crowd_in_semantic=False, dataset=None):

        self.logger = logger
        self.logger.info('ctrl/dataset/gen_panoptic_gt_labels_v2.py --> class PanopticTargetGenerator() : __init__()')

        self.dataset = dataset
        self.ignore_label = ignore_label
        self.rgb2id = rgb2id
        self.thing_list = thing_list
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.ignore_crowd_in_semantic = ignore_crowd_in_semantic

        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, panoptic, segments, mode='train'):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18
        Args:
            panoptic: numpy.array, colored image encoding panoptic label.
            segments: List, a list of dictionary containing information of every segment, it has fields:
                - id: panoptic id, after decoding `panoptic`.
                - category_id: semantic class id.
                - area: segment area.
                - bbox: segment bounding box.
                - iscrowd: crowd region.
        Returns:
            A dictionary with fields:
                - semantic: Tensor, semantic label, shape=(H, W).
                - foreground: Tensor, foreground mask label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(1, H, W).
                - center_points: List, center coordinates, with tuple (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is (offset_y, offset_x).
                - semantic_weights: Tensor, loss weight for semantic prediction, shape=(H, W).
                - center_weights: Tensor, ignore region of center prediction, shape=(H, W), used as weights for center
                    regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
                - offset_weights: Tensor, ignore region of offset prediction, shape=(H, W), used as weights for offset
                    regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
        """
        dn = self.dataset
        panoptic = self.rgb2id(panoptic)
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        instance = np.zeros_like(panoptic, dtype=np.uint8)
        foreground = np.zeros_like(panoptic, dtype=np.uint8)
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(panoptic, dtype=np.float32)
        x_coord = np.ones_like(panoptic, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)

        instance_id = 1
        for seg in segments:
            cat_id = seg["category_id"]

            # TODO:
            # as in the validation json file,
            # the cat_id are the id and not the trainIds,
            # so we need to map the id to trainId
            # because our self.thing_list uses trainIds
            # where as for train json file,
            # the cat_id are the continous trainIds (0,..,15)
            if dn == 'cityscapes' and mode == 'val':
                labelInfo = id2label[cat_id]
                cat_id = labelInfo.trainId

            if self.ignore_crowd_in_semantic:
                if not seg['iscrowd']:
                    semantic[panoptic == seg["id"]] = cat_id
            else:
                semantic[panoptic == seg["id"]] = cat_id

            # this block is required only for cityscapes # TODO
            if cat_id in self.thing_list and dn == 'cityscapes':
                foreground[panoptic == seg["id"]] = 1

            # this block is required only for synthia or source domain # TODO
            if cat_id in self.thing_list: # and dn == 'synthia':
                instance[panoptic == seg["id"]] = instance_id
                instance_id += 1

            if not seg['iscrowd']:
                # Ignored regions are not in `segments`.
                # Handle crowd region.
                center_weights[panoptic == seg["id"]] = 1
                if self.ignore_stuff_in_offset: # ignore_stuff_in_offset = True
                    # Handle stuff region.
                    if cat_id in self.thing_list:
                        offset_weights[panoptic == seg["id"]] = 1
                else:
                    offset_weights[panoptic == seg["id"]] = 1

            '''
            The below if block generates the center and offsets for thing classes.
            There are some segments belonging to thing classes which has iscrowd falg True or 1.
            The below block generates center and offsets for those segments as well,
            but the center_weights and offset_weights generated in the above if block take care of these segments
            by multipying 0 with the center and offset losses,i.e., for these crowd region segments (belong to thing class)
            we dont compute the center and offset losses.            
            '''
            # if cat_id in self.thing_list and dn == 'cityscapes' or cat_id in self.thing_list and dn == 'synthia' and not seg['iscrowd']:
            # ideally we should also put the condition -->  and not seg['iscrowd']: -- but does not matter,those crowd region has offset and center wieghts 0 -- see the above code segment - loss willbe 0 and grad won't backprop
            if cat_id in self.thing_list: # and not seg['iscrowd']:
                # find instance center
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area  # TODO: this block is only required for cityscapes by original panoptic deeplab semantic loss, they use this weight semantic_weights, I am not using it
                if dn == 'cityscapes':
                    ins_area = len(mask_index[0])
                    if ins_area < self.small_instance_area:
                        semantic_weights[panoptic == seg["id"]] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])

                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(
                    center[0, aa:bb, cc:dd], self.g[a:b, c:d])

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]

        if mode == 'train' or mode == 'all':  # train for cityscapes and mapillary and all for synthia
            return dict(
                semantic=semantic.astype('long'),
                center=center.astype(np.float32),
                offset=offset.astype(np.float32),
                center_weights=center_weights.astype(np.float32),
                offset_weights=offset_weights.astype(np.float32),
                instance=instance.astype('long'),
            )
        elif mode == 'val':
            return dict(
                semantic=semantic.astype('long'),
                foreground=foreground.astype('long'),
                center=center.astype(np.float32),
                center_points=center_pts,
                offset=offset.astype(np.float32),
                semantic_weights=semantic_weights.astype(np.float32),
                center_weights=center_weights.astype(np.float32),
                offset_weights=offset_weights.astype(np.float32),
                instance=instance.astype('long') # TODO: comment
            )


class SemanticTargetGenerator(object):
    """
    Generates semantic training target only for Panoptic-DeepLab (no instance).
    Annotation is assumed to have Cityscapes format.
    Arguments:
        ignore_label: Integer, the ignore label for semantic segmentation.
        rgb2id: Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
        thing_list: List, a list of thing classes
        sigma: the sigma for Gaussian kernel.
    """
    def __init__(self, ignore_label, rgb2id):
        self.ignore_label = ignore_label
        self.rgb2id = rgb2id

    def __call__(self, panoptic, segments):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18
        Args:
            panoptic: numpy.array, colored image encoding panoptic label.
            segments: List, a list of dictionary containing information of every segment, it has fields:
                - id: panoptic id, after decoding `panoptic`.
                - category_id: semantic class id.
                - area: segment area.
                - bbox: segment bounding box.
                - iscrowd: crowd region.
        Returns:
            A dictionary with fields:
                - semantic: Tensor, semantic label, shape=(H, W).
        """
        panoptic = self.rgb2id(panoptic)
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        for seg in segments:
            cat_id = seg["category_id"]
            semantic[panoptic == seg["id"]] = cat_id

        return dict(
            semantic=torch.as_tensor(semantic.astype('long'))
        )