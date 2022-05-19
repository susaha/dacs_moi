from ctrl.model_panop.inst_seg import InstSeg

cfg = {}
cfg.POST_PROCESSING = {}
cfg.POST_PROCESSING.CENTER_THRESHOLD = 0.1
cfg.POST_PROCESSING.NMS_KERNEL = 7
cfg.POST_PROCESSING.TOP_K_INSTANCE = 200
# thing_list=data_loader.dataset.cityscapes_thing_list
thing_list=[11,12,13]

inst_seg = InstSeg(cfg, thing_list)

# semantic_pred = get_semantic_segmentation(out_dict['semantic'])
semantic_pred = None
pred_source = {}
pred_source['center'] = None
pred_source['offset'] = None
instance, center = inst_seg(semantic_pred, pred_source['center'], pred_source['offset'])


'''
# chnage the disc in common_configpnop_danda
get the unqie instance ids - insIds = instance.unique()
semanitc predictions : sem
center predcitions heatmap: ctr_hmp
for id in insIds:
    mask = instance == id                               # M: is a H x W matrix
    w = ctr_hmp[mask]                         # H * M : is a H x W matrix
    
    # inst_scores has score values between 0 and 1
    # normalize the score values between 0 and 1        # w = (H * M) / || H * M ||
    
    w = w.view(-1) # flatten a t1 NxW matrix to H*W vector
    w = torch.nn.functional.normalize(w, dim=0)
    w = w.view((H, W))
    
    x = coord_x * w
    x = x.sum()
    y = coord_y * w
    y = y.sum()
    
Another approach:

for imid in num_batches:
    boxes = []
    for id in insIds:
        mask = instance == id                               # M: is a H x W matrix
        
        # the below code is to get the bounding box for each segment
        # you need to convert the numpy to torch 
        area = np.sum(mask) # segment area computation
        
        # bbox computation for a segment
        hor = np.sum(mask, axis=0)
        hor_idx = np.nonzero(hor)[0]
        x = hor_idx[0]
        width = hor_idx[-1] - x + 1
        vert = np.sum(mask, axis=1)
        vert_idx = np.nonzero(vert)[0]
        y = vert_idx[0]
        height = vert_idx[-1] - y + 1
        bbox = [imid, int(x), int(y), int(width), int(height)]
        boxes.append(bbox)

once you have the boxes then do the roi pooling
def roi_pooling(input, rois, size=(7,7), spatial_scale=1.0):
    assert(rois.dim() == 2)
    assert(rois.size(1) == 5)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)
    
    rois[:,1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
        output.append(adaptive_max_pool(im, size))    
    return torch.cat(output, 0)

        
    

    
    
    
    
    
    
    
    
    
    
    
    
'''


