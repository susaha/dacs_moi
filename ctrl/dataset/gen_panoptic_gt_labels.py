# ------------------------------------------------------------------------------
# Generates targets for Panoptic-DeepLab.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import numpy as np
import torch


class PanopticTargetGeneratorForSynthiaForEvaluationOnSynthia(object):
    def __init__(self, logger, ignore_label, rgb2id, thing_list, sigma=8, ignore_stuff_in_offset=False,
                 small_instance_area=0, small_instance_weight=1, ignore_crowd_in_semantic=False):
        self.logger = logger
        self.logger.info('ctrl/dataset/gen_panoptic_gt_labels.py --> class PanopticTargetGeneratorForSynthiaForEvaluationOnSynthia() : __init__()')
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
        self.logger.info('class PanopticTargetGenerator(object): def __init__(..)')

    def __call__(self, panoptic, segments, VIS=False, mode=None, dataset=None):
        panoptic = self.rgb2id(panoptic)
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
        semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
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
                center_weights[panoptic == seg["id"]] = 1
                if self.ignore_stuff_in_offset:
                    if cat_id in self.thing_list:
                        offset_weights[panoptic == seg["id"]] = 1
                else:
                    offset_weights[panoptic == seg["id"]] = 1

            if cat_id in self.thing_list:
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    continue
                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic == seg["id"]] = self.small_instance_weight
                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])
                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary # TODO
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]
                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd], self.g[a:b, c:d])
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]

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

class PanopticTargetGeneratorForSynthia(object):
    def __init__(self, logger, ignore_label, rgb2id, thing_list, sigma=8, ignore_stuff_in_offset=False,
                 small_instance_area=0, small_instance_weight=1, ignore_crowd_in_semantic=False):
        self.logger = logger
        self.logger.info('ctrl/dataset/gen_panoptic_gt_labels.py --> class PanopticTargetGeneratorForSynthia() : __init__()')
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

    def __call__(self, panoptic, segments, VIS=False, mode=None, dataset=None):
        panoptic = self.rgb2id(panoptic)
        height, width = panoptic.shape[0], panoptic.shape[1]
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(panoptic, dtype=np.float32)
        x_coord = np.ones_like(panoptic, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        center_weights = np.ones_like(panoptic, dtype=np.uint8)
        # center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        for seg in segments:
            cat_id = seg["category_id"]
            # center_weights[panoptic == seg["id"]] = 1
            if cat_id in self.thing_list:
                offset_weights[panoptic == seg["id"]] = 1
            if cat_id in self.thing_list:
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    continue
                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])
                y, x = int(center_y), int(center_x)
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]
                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd], self.g[a:b, c:d])
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]
        return dict(
            center=center.astype(np.float32),
            offset=offset.astype(np.float32),
            center_weights=center_weights.astype(np.float32),
            offset_weights=offset_weights.astype(np.float32)
        )


class PanopticTargetGeneratorForSynthiaClassWise(object):
    def __init__(self, logger, ignore_label, rgb2id, thing_list, sigma=8, ignore_stuff_in_offset=False,
                 small_instance_area=0, small_instance_weight=1, ignore_crowd_in_semantic=False):
        self.logger = logger
        self.logger.info('ctrl/dataset/gen_panoptic_gt_labels.py --> class PanopticTargetGeneratorForSynthiaClassWise() : __init__()')

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

    def __call__(self, panoptic, segments, VIS=False):
        panoptic = self.rgb2id(panoptic)
        height, width = panoptic.shape[0], panoptic.shape[1]
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(panoptic, dtype=np.float32)
        x_coord = np.ones_like(panoptic, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        num_thing_classes = len(self.thing_list)
        dcid = {}
        cid_count = 0
        for cls in self.thing_list:
            dcid[cls] = cid_count
            cid_count += 1
        # TODO: the following labels are generated for DACS training with class-mix
        center_cls_wise = np.zeros((num_thing_classes, height, width), dtype=np.float32)
        offset_cls_wise = np.zeros((num_thing_classes, 2, height, width), dtype=np.float32)
        # TODO: as I don't know if Synthia has a iscrowd flag, I am setting all values for center_weight to 1.0
        center_weights_cls_wise = np.ones((num_thing_classes, panoptic.shape[0], panoptic.shape[1]), dtype=np.uint8)
        offset_weights_cls_wise = np.zeros((num_thing_classes, panoptic.shape[0], panoptic.shape[1]), dtype=np.uint8)
        for seg in segments:
            cat_id = seg["category_id"]
            # center_weights_cls_wise[dcid[cat_id]][panoptic == seg["id"]] = 1 # this is not possible , cat_id belongs to both stuff and thing and background 255, dcid[cat_id] has only 6 instances
            if cat_id in self.thing_list:
                offset_weights_cls_wise[dcid[cat_id]][panoptic == seg["id"]] = 1
            if cat_id in self.thing_list:
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue
                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])
                y, x = int(center_y), int(center_x)
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]
                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                # TODO: the following label is generated for DACS training with class-mix
                # center
                center_cls_wise[dcid[cat_id], aa:bb, cc:dd] = np.maximum(center_cls_wise[dcid[cat_id], aa:bb, cc:dd], self.g[a:b, c:d])
                # offset
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]
                # TODO: the following labels are generated for DACS training with class-mix
                offset_y_index_cls_wise = (np.ones_like(mask_index[0])*dcid[cat_id], np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index_cls_wise = (np.ones_like(mask_index[0])*dcid[cat_id], np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_cls_wise[offset_y_index_cls_wise] = center_y - y_coord[mask_index]
                offset_cls_wise[offset_x_index_cls_wise] = center_x - x_coord[mask_index]
        # synthia all: we need these for training on synthia
        return dict(
            center=center_cls_wise,
            offset=offset_cls_wise,
            center_weights=center_weights_cls_wise,
            offset_weights=offset_weights_cls_wise,
        )


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

    def __call__(self, panoptic, segments, VIS=False, mode=None, dataset=None):
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
        # id = synthia semantic class ids after mapping with this formula: id = id*1000 + segment_count

        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)


        for seg in segments:

            cat_id = seg["category_id"]

            # get the semanitc label map
            if self.ignore_crowd_in_semantic: # self.ignore_crowd_in_semantic=False
                if not seg['iscrowd']:
                    semantic[panoptic == seg["id"]] = cat_id
            else:
                semantic[panoptic == seg["id"]] = cat_id

            # get the foreground map for the things, ignoring the stuffs
            if cat_id in self.thing_list:
                foreground[panoptic == seg["id"]] = 1

            # get the center_weights and offset_weights
            # for things  seg['iscrowd']=False, so this condition is true for things and not for stuff #TODO
            if not seg['iscrowd']:
                # Ignored regions are not in `segments`. # Handle crowd region.
                center_weights[panoptic == seg["id"]] = 1

                # self.ignore_stuff_in_offset=True,  # TODO
                if self.ignore_stuff_in_offset:
                    # Handle stuff region.
                    if cat_id in self.thing_list:
                        offset_weights[panoptic == seg["id"]] = 1
                else:
                    offset_weights[panoptic == seg["id"]] = 1


            if cat_id in self.thing_list:

                # mask = (panoptic == seg["id"]) : returns a logical mask of shape height x width ;
                # np.where(mask) returns the column (mask_index[0] denotes y axis) and row (mask_index[1] denotes x axis) values
                # (or y,x coordinate values) where the mask is True
                mask_index = np.where(panoptic == seg["id"])

                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic == seg["id"]] = self.small_instance_weight

                # based on the mask_index which has the (x,y) coordinate values for this segment or instance, # TODO
                # compute the instance or semgment centroid, center_y, center_x are scalar values
                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])

                # generate center heatmap # TODO
                y, x = int(center_y), int(center_x)

                # outside image boundary # TODO
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue

                sigma = self.sigma # sigman is 8 # TODO

                # upper left: compute the upper left cordinate point ul from center (x,y), # TODO
                # note sigma is 8 and 3x8=24, so the Gaussian wndow around the center is of shape 51x51
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))

                # bottom right: compute the bottom right cooridnate point br from center # TODO
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], width) - ul[0] # c:0, d:51 # TODO
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1] # a:0, b:51 # TODO

                cc, dd = max(0, ul[0]), min(br[0], width) # dd:343, cc:292 --> denotes x coordinates or widht # TODO
                aa, bb = max(0, ul[1]), min(br[1], height) # bb:106, aa:55 --> denotes y coordinates or heght # TODO

                # gitting Gaussian of 51x51 window around the center # TODO
                # center is intially a 1xHxW ndarray with all 0 values,
                # center[0, aa:bb, cc:dd]: is a 51x51 ndaaray,
                # self.g[a:b, c:d] or self.g is a 51x51 ndarray - this is the precomputed Gaussian in the init function
                # this Gaussian is used as a GT label for the center
                center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd], self.g[a:b, c:d])

                # generate offset (2, h, w) -> (y-dir, x-dir)
                '''
                mask_index:{tuple:2}
                mask_index[0]: --> y coordinate's positional index: {ndarray:(N)}, int64, where N is the number of pixels in the segment or instance
                mask_index[1]: --> x coordinate's positional index: {ndarray:(N)}, int64, where N is the number of pixels in the segment or instance
                offset_y_index: {tuple:3}
                offset_y_index[0]: ndarray of shape N, with all 0s
                offset_y_index[0]: mask_index[0]
                offset_y_index[1]: mask_index[1]
                offset is of shape 2,H,W, 
                offset_y_index is used to access the elements of offset tensor along first channel that is offset[0,:]
                all elements of offset_y_index[0] are 0s, so it access the first channel of offset, ie. offset[0,:]
                offset_y_index[1] has the y coordinate's positional index as per the mask_index[0], that means in a HxW plane,
                where the instance or segment is present you have ones, and then you genrate a mask_index for the non zero entries
                mask_index[0] has the y coordinates  and , mask_index[1] has the x coordinate
                offset[offset_y_index] stores the distance of each pixel in the segment from the center in y direction
                offset[offset_y_index] = center_y - y_coord[mask_index] --> this line basically compute the 
                distane between the y cordinate for the center , i.e., center_y which is a scalar value 
                and the y coordinates of all the pixels in the segment of instance 
                similary offset[offset_x_index] = center_x - x_coord[mask_index] computes the distance between the cenr_x and all x_coordnates of the otehr pixels
                '''
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]


        if mode == 'train' and dataset == 'cityscapes':
            return dict(
                center=center.astype(np.float32),
                offset=offset.astype(np.float32),
                center_weights=center_weights.astype(np.float32),
                offset_weights=offset_weights.astype(np.float32)
            )
        elif mode == 'val_visualize' and dataset == 'cityscapes':
            return dict(
                semantic_instance=semantic.astype('long'),
                foreground=foreground.astype('long'),
                center=center.astype(np.float32),
                center_points=center_pts,
                offset=offset.astype(np.float32),
                semantic_weights=semantic_weights.astype(np.float32),
                center_weights=center_weights.astype(np.float32),
                offset_weights=offset_weights.astype(np.float32)
            )
        else:
            # original code
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