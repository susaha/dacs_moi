from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class PanopPseudoLabels(nn.Module):

    def __init__(self, cfg, thing_list,):

        super(PanopPseudoLabels, self).__init__()

        print('ctrl/panop_pseudo_labels.py --> class PanopPseudoLabels(...) -->  def __init__(...)')

        self.cfg = cfg
        self.thing_list = thing_list
        self.threshold = cfg.POST_PROCESSING.CENTER_THRESHOLD
        self.nms_kernel = cfg.POST_PROCESSING.NMS_KERNEL
        self.top_k = cfg.POST_PROCESSING.TOP_K_INSTANCE

        self.height = self.cfg.DATASET.RANDOM_CROP_DIM
        self.width = self.cfg.DATASET.RANDOM_CROP_DIM
        offsets_dtype = torch.float32 # TODO: check the dtype of offset tensor (pytroch types: https://pytorch.org/docs/stable/tensor_attributes.html)
        DEVICE = torch.device('cuda:0')
        self.device = DEVICE

        # generating the Gaussian window of 51 x 51 , the mean of the Gaussian is 1
        # the center 25x25 is 1 and the values around center is high e.g.0.9 and 0.8
        # as we move further from the center the values are low e.g. the corner values are almost 0
        self.sigma = 8
        sigma = self.sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.g = torch.from_numpy(self.g).to(DEVICE)

        if self.cfg.GEN_OFFSET_PSEUDO_LABELS:
            # generates a coordinate map, where each location is the coordinate of that loc
            self.y_coord = torch.arange(self.height, dtype=offsets_dtype, device=DEVICE).repeat(1, self.width, 1).transpose(1, 2)
            self.x_coord = torch.arange(self.width, dtype=offsets_dtype, device=DEVICE).repeat(1, self.height, 1)
            self.coord = torch.cat((self.y_coord, self.x_coord), dim=0)

    def _group_pixels_ori(self, ctr, offsets):
        # if offsets.size(0) != 1:
        #     raise ValueError('Only supports inference for batch size = 1')
        ctr_loc = self.coord + offsets
        ctr_loc = ctr_loc.reshape((2, self.height * self.width)).transpose(1, 0)
        # ctr: [K, 2] -> [K, 1, 2]
        # ctr_loc = [H*W, 2] -> [1, H*W, 2]
        ctr = ctr.unsqueeze(1)
        ctr_loc = ctr_loc.unsqueeze(0)
        # distance: [K, H*W]
        distance = torch.norm(ctr - ctr_loc, dim=-1)
        # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
        seg_id_maps = torch.argmin(distance, dim=0).reshape((1, self.height, self.width)) + 1
        return seg_id_maps

    def _group_pixels_without_for_loop(self, ctr_batch, offsets):
        '''
       self.coord: [2,h,w]
       offsets:     [2,2,h,w] [batchSize, xy coord, h, W]
       ctr_loc: [2,2,h,w]
       ctr_batch: [K, 3]
       ctr: [K,2]
        '''
        bs = offsets.shape[0]
        ctr = ctr_batch[:,1:3]
        bids = ctr_batch[:,0].unique().tolist()
        ctr_loc = self.coord + offsets # note coord is same for 2 images, but offsets are differnet for two images
        seg_id_maps = torch.zeros((bs, self.height, self.width)).to(self.device)
        # ctr: [K, 2] -> [K, 1, 2]
        ctr = ctr.unsqueeze(dim=1)
        # ctr: [K, 1, 2] -> [1, K, 1, 2]
        ctr = ctr.unsqueeze(dim=0)
        # ctr_loc: [2,2,h,w] -> [2, h, w, 2]
        ctr_loc = ctr_loc.permute(0, 2, 3, 1)
        # ctr_loc: [2, h, w, 2] -> [2, 1, h*w, 2]
        ctr_loc = ctr_loc.reshape((bs, 1, self.height * self.width, 2))
        # distance: [bs, K, H*W]
        distance = torch.norm(ctr - ctr_loc, dim=-1)
        for bid in bids:
            batch_mask = ctr_batch[:,0] == bid
            dis = distance[bid, batch_mask, :]
            # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
            seg_id_maps[bid, :] = torch.argmin(dis, dim=0).reshape((self.height, self.width)) + 1
        return seg_id_maps


    def _group_pixels_with_for_loop(self, ctr_batch, offsets_batch):
        '''
        ctr_batch.shape = [num_selected_center, 3]
        ctr_batch[:,0] = batchId
        ctr_batch[:,1] = center_y
        ctr_batch[:,2] = center_x
        offsets_batch.shape = [batch_size, 2, H, W], e.g. [2, 2, H, W]
        '''
        batch_size = offsets_batch.shape[0]
        seg_id_maps = torch.zeros((batch_size, self.height, self.width)).to(self.device)
        for bid in range(batch_size):
            ctr = ctr_batch[ctr_batch[:, 0] == bid][:, 1:3]
            ctr_loc = self.coord + offsets_batch[bid, :].unsqueeze(dim=0)
            ctr_loc = ctr_loc.reshape((2, self.height * self.width)).transpose(1, 0)
            # ctr: [K, 2] -> [K, 1, 2]
            # ctr_loc = [H*W, 2] -> [1, H*W, 2]
            ctr = ctr.unsqueeze(1)
            ctr_loc = ctr_loc.unsqueeze(0)
            # distance: [K, H*W]
            distance = torch.norm(ctr - ctr_loc, dim=-1)
            # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
            seg_id_maps[bid, :] = torch.argmin(distance, dim=0).reshape((self.height, self.width)) + 1
        return seg_id_maps

    def _gen_pseudo_labels(self, ctr, center, thing_seg_map, MixMasks):
        '''
            ctr[:, 0] = is the batchIds
            ctr[:, 1] = is the y coord (colids or image height)
            ctr[:, 2] = is the x coord (rowids or image widht)
        '''
        sigma = self.sigma
        ul = ctr[:, 1:3] - 3 * sigma - 1  # upper left corner point: ul[0] -> y coord or height, ul[1] -> x coord or width
        br = ctr[:, 1:3] + 3 * sigma + 2  # bottom right corner point: br[0] -> y coord or height, br[1] -> x coord or width
        num_centers = ctr.shape[0]
        num_valid_centers = 0
        selected_centers = []
        for i in range(num_centers):
            batchId = ctr[i, 0].item()
            cy = ctr[i, 1].item()   # TODO
            cx = ctr[i, 2].item()  # TODO
            if cx < 0 or cy < 0 or cx >= self.width or cy >= self.height:
                continue
            # TODO: in mixmask, those locations (y,x) values are 1 where source image pixels are pasted
            # if the center (cx,cy) falls in those location,then we don't want to generate the center pseduo label
            # because in the augmented image,in that location source image pixels and gt cener labels are present
            if MixMasks[batchId][:,cy,cx] == 1:
                continue

            a, b = max(0, -ul[i, 0]), min(br[i, 0], self.height) - ul[i, 0]
            c, d = max(0, -ul[i, 1]), min(br[i, 1], self.width) - ul[i, 1]
            aa, bb = max(0, ul[i, 0]), min(br[i, 0], self.height)
            cc, dd = max(0, ul[i, 1]), min(br[i, 1], self.width)

            if thing_seg_map[batchId, aa:bb, cc:dd].sum() >= self.cfg.CENTER_PSEUDO_LBL_MIN_PXL:
                if self.cfg.DEBUG:
                    # print('{} number of pixels labeled as things for image {}'.format(thing_seg_map[batchId, aa:bb, cc:dd].sum(), batchId+1))
                    num_valid_centers += 1
                # and update the center pseudo label with a Gaussian around it
                center[batchId, 0, aa:bb, cc:dd] = torch.maximum(center[batchId, 0, aa:bb, cc:dd], self.g[a:b, c:d])
                selected_centers.append([batchId, cy, cx])
        # if self.cfg.DEBUG:
            # print('number of valid centers used as pseduo labels: {}'.format(num_valid_centers))

        # center is a pseudo label center heatmap of shape BS,1,H,W
        # selected_centers is a tensor of [shape num_valid_centers x 3]
        selected_centers = torch.LongTensor(selected_centers).to(self.device)
        return center, selected_centers


    def _gen_pseudo_labels_wrong_version(self, ctr, center, thing_seg_map, MixMasks):
        '''
        # ctr is a tensor of shape [num_centers, 3] ,
        # each row in ctr denotes a center point,
        # each row has 3 values,
        # first one denote the batchId (0 or 1 for batchSize=2),
        # second one denotes x coordinate value,
        # 3rd one y coordinate value.
        '''
        # TODO: for a center x,y in ctr,check if thing_seg_map[x,y] == 1, then only consider this as a valid center prediction
        sigma = self.sigma
        # ctr.shape [num_center_points, 3], where num_ctr_points is the total number of center points all the images in the batch
        # e.g. if the bacthSize is 2, and one image has 3 predicted center and other one has 5,then the num_ctr_points=8
        # ctr[:, 0] = is the batchIds
        # ctr[:, 1] = is the x coord
        # ctr[:, 2] = is the y coord
        # upper right
        # ul:[num_ctr_points, 2]; ul[:,0] -> x1, ul[:,1] -> y1
        ul = ctr[:, 1:3] - 3 * sigma - 1
        # bottom right
        # br:[num_ctr_points, 2]; br[:,0] -> x2, br[:,1] -> y2
        br = ctr[:, 1:3] + 3 * sigma + 2
        num_centers = ctr.shape[0]
        # TODO: optimize this part by replacing the for loop
        # for each center, fit a Gaussian of 51 x 51 arounf it
        num_valid_centers = 0
        selected_centers = []
        for i in range(num_centers):
            batchId = ctr[i, 0].item()
            cx = ctr[i, 1].item() # TODO
            cy = ctr[i, 2].item()   # TODO
            if cx < 0 or cy < 0 or cx >= self.width or cy >= self.height:
                continue
            if MixMasks[batchId][:,cy,cx] == 1:
                continue
            c, d = max(0, -ul[i, 0]), min(br[i, 0], self.width) - ul[i, 0]
            a, b = max(0, -ul[i, 1]), min(br[i, 1], self.height) - ul[i, 1]
            cc, dd = max(0, ul[i, 0]), min(br[i, 0], self.width)
            aa, bb = max(0, ul[i, 1]), min(br[i, 1], self.height)
            # check the predicted semantic labels of the pixels in the 51x51 window around the predicted center (cx,cy)
            # if at least 100 pixel lables belong to any of the 6 thing classes then consider this center as a valid center
            if thing_seg_map[batchId, aa:bb, cc:dd].sum() >= self.cfg.CENTER_PSEUDO_LBL_MIN_PXL:
                if self.cfg.DEBUG:
                    # print('{} number of pixels labeled as things for image {}'.format(thing_seg_map[batchId, aa:bb, cc:dd].sum(), batchId+1))
                    num_valid_centers += 1
                # and update the center pseudo label with a Gaussian around it
                center[batchId, 0, aa:bb, cc:dd] = self.g[a:b, c:d]
                selected_centers.append([batchId, cy,cx])
        # if self.cfg.DEBUG:
            # print('number of valid centers used as pseduo labels: {}'.format(num_valid_centers))

        # center is a pseudo label center heatmap of shape BS,1,H,W
        # selected_centers is a tensor of [shape num_valid_centers x 3]
        selected_centers = torch.LongTensor(selected_centers).to(self.device)
        return center, selected_centers

        # ul = int(torch.round(ctr[:, 1] - 3 * sigma - 1)), int(torch.round(ctr[:, 2] - 3 * sigma - 1))
        # br = int(torch.round(ctr[:, 1] + 3 * sigma + 2)), int(torch.round(ctr[:, 2] + 3 * sigma + 2))
        # c, d = max(0, -ul[0]), min(br[0], width) - ul[0]  # c:0, d:51 # TODO
        # a, b = max(0, -ul[1]), min(br[1], height) - ul[1]  # a:0, b:51 # TODO
        # cc, dd = max(0, ul[0]), min(br[0], width)  # dd:343, cc:292 --> denotes x coordinates or widht # TODO
        # aa, bb = max(0, ul[1]), min(br[1], height)  # bb:106, aa:55 --> denotes y coordinates or heght # TODO
        # center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd], self.g[a:b, c:d])
        # print()

    # def _get_instance_segmentation(self, sem_seg, ctr_hmp, offsets):
    #     # gets foreground segmentation
    #     thing_seg_map = torch.zeros_like(sem_seg)
    #     for thing_class in self.thing_list:
    #         thing_seg_map[sem_seg == thing_class] = 1
    #     ctr = self._find_instance_center(ctr_hmp)
    #     # if there is no high confident center prediction then return zero tensor
    #     if ctr.size(0) == 0:
    #         return torch.zeros_like(sem_seg), ctr.unsqueeze(0)
    #     ins_seg = self._group_pixels(ctr, offsets)
    #     return thing_seg_map * ins_seg, ctr.unsqueeze(0)

    def _find_instance_center(self, ctr_hmp):

        # thresholding, setting values below threshold to -1
        # ctr_hmp: [BS, 1, H, W] [2,1,500,500]
        ctr_hmp = F.threshold(ctr_hmp, self.threshold, -1)

        # NMS
        nms_padding = (self.nms_kernel - 1) // 2
        ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=self.nms_kernel, stride=1, padding=nms_padding)
        ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1
        # ctr_hmp: [BS, 1, H, W] -> [BS, H, W]
        ctr_hmp = ctr_hmp.squeeze(dim=1)

        # this returns a tensor of shape [num_centers, 3]
        # the first dim denotes batchIds, second dim is column ids (or img height),and the 3rd dim is rowid (img widht)
        return torch.nonzero(ctr_hmp >= self.cfg.THRESHOLD_CENTER)

    def get_instance_pseudo_labels(self, semantic_pred, ctr_hmp, offsets, MixMasks, DEVICE):
        batch_size, _, height, width = ctr_hmp.shape
        center_pseudo_labels = torch.zeros_like(ctr_hmp)  # center_pseudo_labels [2, 1, H, W]
        offset_pseudo_label = torch.zeros_like(offsets)
        center_weights = torch.ones((batch_size, height, width)).byte().to(DEVICE)
        offset_weights = torch.zeros((batch_size, height, width)).byte().to(DEVICE)

        # get centers which have score >=0.9
        # ctr is a tensor of shape [num_centers, 3] ,
        # each row in ctr denotes a center point,
        # each row has 3 values,
        # first one denote the batchId (0 or 1 for batchSize=2),
        # second one denotes x corodinate value,
        # 3rd one y coordinate value
        ctr = self._find_instance_center(ctr_hmp)

        if ctr.shape[0] == 0:
            return center_pseudo_labels, offset_pseudo_label, center_weights, offset_weights

        # gets foreground segmentation for the things
        # if a center belongs to a thing class,then only it is a valid center
        # this way we guide the center prediction using our pseudo semnatic labels
        thing_seg_map = torch.zeros_like(semantic_pred)
        for thing_class in self.thing_list:
            thing_seg_map[semantic_pred == thing_class] = 1

        # gen pseudo labels
        # center [2, 1, H, W] [batchSie, 1, H, W]
        if not self.cfg.GEN_PSEDUO_LABEL_WRONG_VERSION:
            center_pseudo_labels, selected_centers = self._gen_pseudo_labels(ctr, center_pseudo_labels, thing_seg_map, MixMasks)
        else:
            center_pseudo_labels, selected_centers = self._gen_pseudo_labels_wrong_version(ctr, center_pseudo_labels, thing_seg_map, MixMasks)

        if selected_centers.shape[0] == 0:
            return center_pseudo_labels, offset_pseudo_label, center_weights, offset_weights

        if self.cfg.GEN_OFFSET_PSEUDO_LABELS:
            # get the instance segmentation map where each segment (group of pixels) is denoted by an instance ids 1,2, ..., N
            # where N is the number of center points in selected_centers, note that initially there are fewer instance ids
            # note, ctr has centers after thresholding the score, but selected_centers has lesser centers than ctr
            # as center points are filtered out based on the semanitc prediction labels for thing classes
            # selected_centers.shape [num_valid_centers, 3]
            # selected_centers[:,0] -> batchId
            # selected_centers[:,1] -> y
            # selected_centers[:,2] -> x

            # TODO: if you want to visualize  thing_seg_map, seg_id_maps, ins_seg_id_maps then activate this below block
            # TODO: and deactivate the  block after below block and
            # TODO: and then activate and deactivate accrosingly the last line where you return values
            # if not self.cfg.GROUP_PXL_WITH_FORLOOP:
            #     seg_id_maps = self._group_pixels_without_for_loop(selected_centers, offsets)
            # else:
            #     seg_id_maps = self._group_pixels_with_for_loop(selected_centers, offsets)
            # ins_seg_id_maps = thing_seg_map * seg_id_maps

            # TODO: if you want to visualize  thing_seg_map, seg_id_maps, ins_seg_id_maps then deactivate this below block
            # TODO: and activate the above block
            # TODO: and then activate and deactivate accrosingly the last line where you return values
            if not self.cfg.GROUP_PXL_WITH_FORLOOP:
                ins_seg_id_maps = self._group_pixels_without_for_loop(selected_centers, offsets)
            else:
                ins_seg_id_maps = self._group_pixels_with_for_loop(selected_centers, offsets)
            ins_seg_id_maps = thing_seg_map * ins_seg_id_maps

            y_coord = self.y_coord.squeeze(dim=0)
            x_coord = self.x_coord.squeeze(dim=0)
            for bid in range(batch_size):
                inst_ids = ins_seg_id_maps[bid, :].unique()
                for inst_id in inst_ids:
                    # if inst_id == 2:
                    if inst_id != 0:
                        # center_weights[bid][ins_seg_id_maps[bid, :] == inst_id] = 1
                        offset_weights[bid][ins_seg_id_maps[bid, :] == inst_id] = 1
                        # center_weights[bid][MixMasks[bid][0] == 1] = 0
                        # offset_weights[bid][MixMasks[bid][0] == 1] = 0

                        mask_index = torch.where(ins_seg_id_maps[bid, :] == inst_id)
                        center_y, center_x = torch.mean(mask_index[0].float()), torch.mean(mask_index[1].float())
                        if mask_index[0].shape[0] == 0:
                            # the instance is completely cropped
                            continue
                        # generate offset (2, h, w) -> (y-dir, x-dir)
                        offset_y_index = (torch.ones_like(mask_index[0])*bid, torch.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                        offset_x_index = (torch.ones_like(mask_index[0])*bid, torch.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                        offset_pseudo_label[offset_y_index] = center_y - y_coord[mask_index]
                        offset_pseudo_label[offset_x_index] = center_x - x_coord[mask_index]
                        offset_pseudo_label[bid, :, MixMasks[bid][0] == 1] = 0

        # TODO: if you want to visualize  thing_seg_map, seg_id_maps, ins_seg_id_maps then deactivate the following line
        # TODO: and deactivate the line after the following line
        # return center_pseudo_labels, offset_pseudo_label, thing_seg_map, seg_id_maps, ins_seg_id_maps


        return center_pseudo_labels, offset_pseudo_label, center_weights, offset_weights







def main():
    print()


if __name__ == "__main__":
    main()