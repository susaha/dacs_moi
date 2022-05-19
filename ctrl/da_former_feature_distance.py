import numpy as np
import torch
import torch.nn.functional as F


def downscale_label_ratio(gt, scale_factor,  min_ratio, n_classes, ignore_index=255):
    assert scale_factor > 1
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()  # otw. next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(
        out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out


def masked_feat_dist(f1, f2, mask=None):
    feat_diff = f1 - f2
    # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
    pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
    # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
    if mask is not None:
        # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
        pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
        # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
    return torch.mean(pw_feat_dist)


def calc_feat_dist(img, gt, imgnet_backbone, fdist_classes, feat=None, fdist_scale_min_ratio=0.75, num_classes=16, fdist_lambda=0.0005):

    with torch.no_grad():
        feat_imnet = imgnet_backbone(img)
        feat_imnet = [feat_imnet]
        feat_imnet = [f.detach() for f in feat_imnet]
    lay = 0

    # resizing the source feat from NxCx65x65 to NxCx64x64, because 512 is not divisble by 65 and 512/65 = 7.87,
    # we need a feature map spatial dim which is divisble by the original crop size,
    # for DAFormer the feature map spaital dim is 16x16, and 512 is divisble  bz 16
    # I am using DADA resent backbone which outputs 65x65 dim given 512x512 input img
    dim_feat = feat[lay].shape[-1] - 1
    assert feat[lay].shape[-1] == feat_imnet[lay].shape[-1]
    feat_temp = []
    for f in feat:
        feat_temp.append(F.interpolate(f, size=(dim_feat, dim_feat), mode='bilinear', align_corners=True))
    feat = feat_temp
    # resizing the imnet feat from NxCx65x65 to NxCx64x64
    feat_imnet_temp = []
    for f in feat_imnet:
        feat_imnet_temp.append(F.interpolate(f, size=(dim_feat, dim_feat), mode='bilinear', align_corners=True))
    feat_imnet = feat_imnet_temp

    if fdist_classes is not None:
        fdclasses = torch.tensor(fdist_classes, device=gt.device)
        scale_factor = gt.shape[-1] // feat[lay].shape[-1]
        gt_rescaled = downscale_label_ratio(gt, scale_factor, fdist_scale_min_ratio,  num_classes, 255).long().detach()
        fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
        feat_dist = masked_feat_dist(feat[lay], feat_imnet[lay], fdist_mask)
        debug_fdist_mask = fdist_mask
        debug_gt_rescale = gt_rescaled
    else:
        feat_dist = masked_feat_dist(feat[lay], feat_imnet[lay])
    feat_loss = fdist_lambda * feat_dist
    return feat_loss, debug_fdist_mask, debug_gt_rescale



def calc_feat_dist_ori(img, gt, imgnet_backbone, feat=None):
    fdist_classes = [6,7,10,11,12,13,14,15]
    fdist_scale_min_ratio = 0.75
    num_classes = 16
    fdist_lambda = 0.005
    with torch.no_grad():
        feat_imnet = imgnet_backbone(img)
        feat_imnet = [feat_imnet]
        feat_imnet = [f.detach() for f in feat_imnet]
    lay = 0
    dim_feat = feat[lay].shape[-1] - 1
    if fdist_classes is not None:
        fdclasses = torch.tensor(fdist_classes, device=gt.device)
        scale_factor = gt.shape[-1] // feat[lay].shape[-1]
        gt_rescaled = downscale_label_ratio(gt, scale_factor, fdist_scale_min_ratio,  num_classes, 255).long().detach()
        fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
        feat_dist = masked_feat_dist(feat[lay], feat_imnet[lay], fdist_mask)
        debug_fdist_mask = fdist_mask
        debug_gt_rescale = gt_rescaled
    else:
        feat_dist = masked_feat_dist(feat[lay], feat_imnet[lay])
    feat_loss = fdist_lambda * feat_dist
    return feat_loss, debug_fdist_mask, debug_gt_rescale