from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

'''
ctrl/evaluate_panop.py
        semantic_pred = get_semantic_segmentation(out_dict['semantic'])
         if config.TEST.EVAL_INSTANCE or config.TEST.EVAL_PANOPTIC:
                        panoptic_pred, center_pred = get_panoptic_segmentation(
                            semantic_pred,
                            out_dict['center'],
                            out_dict['offset'],
                            thing_list=data_loader.dataset.cityscapes_thing_list,
                            label_divisor=data_loader.dataset.label_divisor,
                            stuff_area=config.POST_PROCESSING.STUFF_AREA,
                            void_label=(
                                    data_loader.dataset.label_divisor *
                                    data_loader.dataset.ignore_label),
                            threshold=config.POST_PROCESSING.CENTER_THRESHOLD,
                            nms_kernel=config.POST_PROCESSING.NMS_KERNEL,
                            top_k=config.POST_PROCESSING.TOP_K_INSTANCE,
                            foreground_mask=foreground_pred)
def get_panoptic_segmentation(sem, ctr_hmp, offsets, thing_list, label_divisor, stuff_area, void_label, threshold=0.1, nms_kernel=3, top_k=None, foreground_mask=None):

TODO:
debug with batch size 1 and 2
'''


class InstSeg(nn.Module):
    '''
    thing_list=data_loader.dataset.cityscapes_thing_list
    '''
    def __init__(self, cfg, thing_list):
        super(InstSeg, self).__init__()

        print('ctrl/model_panop/inst_seg.py --> class InstSeg(...) -->  def __init__(...)')
        self.thing_list = thing_list
        self.threshold = cfg.POST_PROCESSING.CENTER_THRESHOLD
        self.nms_kernel = cfg.POST_PROCESSING.NMS_KERNEL
        self.top_k = cfg.POST_PROCESSING.TOP_K_INSTANCE

    def _group_pixels(self, ctr, offsets):

        # if offsets.size(0) != 1:
        #     raise ValueError('Only supports inference for batch size = 1')

        offsets = offsets.squeeze(0)
        height, width = offsets.size()[1:]

        # generates a coordinate map, where each location is the coordinate of that loc
        y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
        x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
        coord = torch.cat((y_coord, x_coord), dim=0)

        ctr_loc = coord + offsets
        ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)

        # ctr: [K, 2] -> [K, 1, 2]
        # ctr_loc = [H*W, 2] -> [1, H*W, 2]
        ctr = ctr.unsqueeze(1)
        ctr_loc = ctr_loc.unsqueeze(0)

        # distance: [K, H*W]
        distance = torch.norm(ctr - ctr_loc, dim=-1)

        # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
        instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
        return instance_id

    def _find_instance_center(self, ctr_hmp):
        # thresholding, setting values below threshold to -1
        ctr_hmp = F.threshold(ctr_hmp, self.threshold, -1)

        # NMS
        nms_padding = (self.nms_kernel - 1) // 2
        ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=self.nms_kernel, stride=1, padding=nms_padding)
        ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

        # squeeze first two dimensions
        ctr_hmp = ctr_hmp.squeeze()
        assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'

        # find non-zero elements
        ctr_all = torch.nonzero(ctr_hmp > 0)
        if self.top_k is None:
            return ctr_all
        elif ctr_all.size(0) < self.top_k:
            return ctr_all
        else:
            # find top k centers.
            top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), self.top_k)
            return torch.nonzero(ctr_hmp > top_k_scores[-1])

    def _get_instance_segmentation(self, sem_seg, ctr_hmp, offsets):
        # gets foreground segmentation
        thing_seg = torch.zeros_like(sem_seg)
        for thing_class in self.thing_list:
            thing_seg[sem_seg == thing_class] = 1

        ctr = self._find_instance_center(ctr_hmp)
        if ctr.size(0) == 0:
            return torch.zeros_like(sem_seg), ctr.unsqueeze(0)
        ins_seg = self._group_pixels(ctr, offsets)
        return thing_seg * ins_seg, ctr.unsqueeze(0)

    def forward(self, semantic_pred, ctr_hmp, offsets):
        '''
        semantic_pred = get_semantic_segmentation(out_dict['semantic'])
        '''
        instance, center = self._get_instance_segmentation(semantic_pred, ctr_hmp, offsets)
        return instance, center



def main():
    print()


if __name__ == "__main__":
    main()