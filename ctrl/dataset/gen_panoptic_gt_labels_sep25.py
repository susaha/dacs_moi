# ------------------------------------------------------------------------------
# Generates targets for Panoptic-DeepLab.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import numpy as np
import torch


class PanopticTargetGenerator(object):
    """
    Generates panoptic training target for Panoptic-DeepLab.
    Annotation is assumed to have Cityscapes format.
    Arguments:
        ignore_label: Integer, the ignore label for semantic segmentation.
        rgb2id: Function, panoptic label is encoded in a colored image, this function convert color to the corresponding panoptic label.
        thing_list: List, a list of thing classes
        sigma: the sigma for Gaussian kernel.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
        ignore_crowd_in_semantic: Boolean, whether to ignore crowd region in semantic segmentation branch,
            crowd region is ignored in the original TensorFlow implementation.
    """
    def __init__(self, logger, ignore_label, rgb2id, thing_list, sigma=8, ignore_stuff_in_offset=False,
                 small_instance_area=0, small_instance_weight=1, ignore_crowd_in_semantic=False):

        self.logger = logger
        self.logger.info('ctrl/dataset/gen_panoptic_gt_labels.py --> class PanopticTargetGenerator() : __init__()')

        self.ignore_label = ignore_label
        self.rgb2id = rgb2id
        self.thing_list = thing_list
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.ignore_crowd_in_semantic = ignore_crowd_in_semantic

        '''        
        We represent each object instance by its center of mass.
        For every foreground pixel (i.e., pixel whose class is a ‘thing’), we further predict the offset to its corresponding mass center.
        During training, groundtruth instance centers are encoded by a 2-D Gaussian with standard deviation of 8 pixels [panopitc-deeplab cvpr2020]
        '''
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.logger.info('class PanopticTargetGenerator(object): def __init__(..)')

    def __call__(self, panoptic, segments, VIS=False):
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
        panoptic = self.rgb2id(panoptic)
        # uids = np.unique(panoptic)
        # for uid in uids:
        #     self.logger.info('{}'.format(int(uid)), end=' ')
        # self.logger.info()
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
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

        # category_id = continuous train_ids - 0,1,2,...15,255
        # id = cityscapes semantic class ids after mapping with this formula: id = id*1000 + segment_count
        # for example if the cityscape original id is 24 and then the instance id = 0, then the id = 24001, for instance 5 = 24005
        # panoptic has the mapped ids, e.g., 24000, 24001 ... etc.
        # and seg['id'] is basically the segmentInfo saved in the json file, that also have  mapped ids, e.g., 24000, 24001 ... etc.

        # *** NOTE: panoptic == seg["id"] --> this gives you the mask for one segment or instance
        # you loop over one instance/segment at a time using seg["id"]

        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        for seg in segments:
            cat_id = seg["category_id"]
            if self.ignore_crowd_in_semantic:
                if not seg['iscrowd']:
                    semantic[panoptic == seg["id"]] = cat_id
            else:
                semantic[panoptic == seg["id"]] = cat_id
            if cat_id in self.thing_list:
                foreground[panoptic == seg["id"]] = 1
            if not seg['iscrowd']:
                # Ignored regions are not in `segments`.
                # Handle crowd region.
                center_weights[panoptic == seg["id"]] = 1
                if self.ignore_stuff_in_offset:
                    # Handle stuff region.
                    if cat_id in self.thing_list:
                        offset_weights[panoptic == seg["id"]] = 1
                else:
                    offset_weights[panoptic == seg["id"]] = 1
            if cat_id in self.thing_list:
                # find instance center
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
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


        if VIS:
            import os
            from PIL import Image
            out_path = '/media/suman/CVLHDD/apps/experiments/CVPR2022/cvpr2022/debug/visualization/synthia-panoptic-target-gt-visuals-1280x760-Aug23'
            f1 = os.path.join(out_path, 'semantic.png')
            im = Image.fromarray(semantic * 100)
            im = im.convert('RGB')
            im.save(f1)
            f1 = os.path.join(out_path, 'foreground.png')
            im = Image.fromarray(foreground * 100)
            im = im.convert('RGB')
            im.save(f1)
            f1 = os.path.join(out_path, 'center.png')
            im = Image.fromarray(center.squeeze() * 100)
            im = im.convert('RGB')
            im.save(f1)
            f1 = os.path.join(out_path, 'offset_x.png')
            im = Image.fromarray(offset[0,:,:] * -1)
            im = im.convert('RGB')
            im.save(f1)
            f1 = os.path.join(out_path, 'offset_y.png')
            im = Image.fromarray(offset[1, :, :] * -1)
            im = im.convert('RGB')
            im.save(f1)
            f1 = os.path.join(out_path, 'semantic_weights.png')
            im = Image.fromarray(semantic_weights * 50)
            im = im.convert('RGB')
            im.save(f1)
            f1 = os.path.join(out_path, 'center_weights.png')
            im = Image.fromarray(center_weights * 100)
            im = im.convert('RGB')
            im.save(f1)
            f1 = os.path.join(out_path, 'offset_weights.png')
            im = Image.fromarray(offset_weights * 100)
            im = im.convert('RGB')
            im.save(f1)
            # self.logger.info()

        return dict(
            semantic_instance=torch.as_tensor(semantic.astype('long')),
            foreground=torch.as_tensor(foreground.astype('long')),
            center=torch.as_tensor(center.astype(np.float32)),
            center_points=center_pts,
            offset=torch.as_tensor(offset.astype(np.float32)),
            semantic_weights=torch.as_tensor(semantic_weights.astype(np.float32)),
            center_weights=torch.as_tensor(center_weights.astype(np.float32)),
            offset_weights=torch.as_tensor(offset_weights.astype(np.float32))
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