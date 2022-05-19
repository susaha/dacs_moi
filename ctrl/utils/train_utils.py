import torch
from pathlib import Path
import os
from torchvision.utils import make_grid
import torch.nn.functional as F
from ctrl.utils.viz_segmask import colorize_mask
import numpy as np
from ctrl.utils.panoptic_deeplab.flow_vis import flow_compute_color
from ctrl.utils.panoptic_deeplab.save_annotation import label_to_color_image
import logging
import numpy as np
from PIL import Image

def print_output_paths(cfg, is_isl_training=None):
    print('*** output paths ***')
    print('cfg.TRAIN.SNAPSHOT_DIR: {}'.format(cfg.TRAIN.SNAPSHOT_DIR))
    print('cfg.TRAIN.LOG_DIR: {}'.format(cfg.TRAIN.LOG_DIR))
    print('cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL: {}'.format(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL))
    print('cfg.TRAIN_LOG_FNAME: {}'.format(cfg.TRAIN_LOG_FNAME))
    print('cfg.TRAIN.TENSORBOARD_LOGDIR: {}'.format(cfg.TRAIN.TENSORBOARD_LOGDIR))
    print('cfg.TEST.VISUAL_RESULTS_DIR: {}'.format(cfg.TEST.VISUAL_RESULTS_DIR))
    if is_isl_training:
        print('cfg.TRAIN.PSEUDO_LABELS_DIR: {}'.format(cfg.TRAIN.PSEUDO_LABELS_DIR))
    print()


def log_output_paths(cfg, is_isl_training=None):

    logger = logging.getLogger(__name__)

    logger.info('*** output paths ***')
    logger.info('cfg.TRAIN.SNAPSHOT_DIR: {}'.format(cfg.TRAIN.SNAPSHOT_DIR))
    logger.info('cfg.TRAIN.LOG_DIR: {}'.format(cfg.TRAIN.LOG_DIR))
    logger.info('cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL: {}'.format(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL))
    logger.info('cfg.TRAIN_LOG_FNAME: {}'.format(cfg.TRAIN_LOG_FNAME))
    logger.info('cfg.TRAIN.TENSORBOARD_LOGDIR: {}'.format(cfg.TRAIN.TENSORBOARD_LOGDIR))
    logger.info('cfg.TEST.VISUAL_RESULTS_DIR: {}'.format(cfg.TEST.VISUAL_RESULTS_DIR))
    logger.info('cfg.TRAIN.DACS_VISUAL_RESULTS_DIR: {}'.format(cfg.TRAIN.DACS_VISUAL_RESULTS_DIR))
    if is_isl_training:
        logger.info('cfg.TRAIN.PSEUDO_LABELS_DIR: {}'.format(cfg.TRAIN.PSEUDO_LABELS_DIR))
    logger.info('***')


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate, DEBUG):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITER, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    num_param_grps = len(optimizer.param_groups)
    if num_param_grps > 1:
        for i in range(1, num_param_grps):
            optimizer.param_groups[i]['lr'] = lr * 10
    return lr


def adjust_learning_rate(optimizer, i_iter, cfg, DEBUG):
    return _adjust_learning_rate(optimizer, i_iter, cfg, cfg.SOLVER.BASE_LR, DEBUG=DEBUG)
    # return _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE, DEBUG=DEBUG) # original dada code


def adjust_learning_rate_disc(optimizer, i_iter, cfg, DEBUG):
    return _adjust_learning_rate(optimizer, i_iter, cfg, cfg.SOLVER.DISC_LR, DEBUG=DEBUG)
    # return _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D, DEBUG=DEBUG) # original dada code


def print_losses(current_losses, i_iter, logger):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.6f} ')
    full_string = ' '.join(list_strings)
    logger.info(f'iter = {i_iter} {full_string}')


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def get_checkpoint_path(i_iter, cfg, current_epoch):
    snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
    bestmodel_dir = Path(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL)
    checkpoint_path = snapshot_dir / f"model_{i_iter}_{current_epoch}.pth"
    bestmodel_path = bestmodel_dir / f"model_{i_iter}_{current_epoch}.pth"
    checkpoint_path_tmp = snapshot_dir / f"model_{i_iter}_{current_epoch}.pth.tmp"
    return checkpoint_path, bestmodel_path, checkpoint_path_tmp


def save_checkpoint(i_iter, cfg, save_dict, checkpoint_path, checkpoint_path_tmp):
    snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
    if i_iter >= cfg.TRAIN.SAVE_PRED_EVERY * 2 and i_iter != 0:
        cp_list = [f for f in os.listdir(str(snapshot_dir)) if 'pth' in f]
        cp_list.sort(reverse=True)
        for f in cp_list:
            checkpoint_path_2_remove = os.path.join(str(snapshot_dir), f)
            strCmd2 = 'rm' + ' ' + checkpoint_path_2_remove
            print('Removing: {}'.format(strCmd2))
            os.system(strCmd2)
    print("Saving the checkpoint as tmp file at: {}".format(checkpoint_path_tmp))
    torch.save(save_dict, checkpoint_path_tmp)
    print("Moving the tmp checkpoint to actual checkpoint at: {}".format(checkpoint_path))
    strCmd = 'mv' + ' ' + str(checkpoint_path_tmp) + ' ' + str(checkpoint_path)
    os.system(strCmd)


def log_losses_tensorboard_cvpr2021(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def log_losses_tensorboard_cvpr2022(writer, loss_meter_dict, i_iter):
    for key in loss_meter_dict.keys():
        # writer.add_scalar('data-actual/{}'.format(key), np.float32(loss_meter_dict[key].val), i_iter)
        writer.add_scalar('data-avg/{}'.format(key), np. float32(loss_meter_dict[key].avg), i_iter)
        # msg += '{name}: {meter.val:.9f} ({meter.avg:.9f})\t'.format(name=key, meter=loss_meter_dict[key])


def draw_in_tensorboard_val(writer, images, i_iter, semantic_pred, pan_to_sem,  pan_to_ins, panoptic_pred, num_classes, label_panop_dict, dataset, tbc):

    height = 256
    width = 512
    permute = [1, 0, 2]
    imgs = []
    color_map = dataset.create_label_colormap(num_classes)
    imgid = tbc
    # image_target
    img_target = images[:3].clone().cpu().data
    # img_target = F.interpolate(img_target, size=(height, width), mode='bilinear', align_corners=False)
    # img_target = img_target.squeeze(dim=0).byte()
    img_target = img_target.squeeze(dim=0)
    imgs.append(img_target)
    strTag = '{} : target_input_image'.format(imgid * 10 + 1)
    num_imgs_per_row = 1
    grid_image = make_grid(imgs, num_imgs_per_row, normalize=True)
    writer.add_image(strTag, grid_image, i_iter)

    imgs = []
    # save gt semantic
    gt_sem = label_panop_dict['semantic'][0].clone().cpu().numpy()
    gt_sem = label_to_color_image(gt_sem, color_map)
    gt_sem = torch.from_numpy(gt_sem.transpose(2, 0, 1))
    gt_sem = gt_sem.unsqueeze(dim=0).float()
    gt_sem = F.interpolate(gt_sem, size=(height, width), mode='bilinear', align_corners=False)
    gt_sem = gt_sem.squeeze(dim=0).byte()
    imgs.append(gt_sem)

    # save semantic_pred
    semPred = np.copy(semantic_pred)
    semPred = label_to_color_image(semPred, color_map)
    semPred = torch.from_numpy(semPred.transpose(2, 0, 1))
    semPred = semPred.unsqueeze(dim=0).float()
    semPred = F.interpolate(semPred, size=(height, width), mode='bilinear', align_corners=False)
    semPred = semPred.squeeze(dim=0).byte()
    imgs.append(semPred)

    strTag = '{} : GT-Semantics : Predicted-Semantics'.format(imgid * 10 + 2)
    num_imgs_per_row = 2
    grid_image = make_grid(imgs, num_imgs_per_row, normalize=False)
    writer.add_image(strTag, grid_image, i_iter)


    imgs = []
    # save pan_to_sem
    pan2sem = np.copy(pan_to_sem)
    pan2sem = label_to_color_image(pan2sem, color_map)
    pan2sem = torch.from_numpy(pan2sem.transpose(2, 0, 1))
    pan2sem = pan2sem.unsqueeze(dim=0).float()
    pan2sem = F.interpolate(pan2sem, size=(height, width), mode='bilinear', align_corners=False)
    pan2sem = pan2sem.squeeze(dim=0).byte()
    imgs.append(pan2sem)

    # save pan_to_ins
    stuff_id = 0
    from ctrl.utils.panoptic_deeplab.save_annotation import random_color
    label = pan_to_ins.copy()
    ids = np.unique(label)
    num_colors = len(ids)
    colormap = np.zeros((num_colors, 3), dtype=np.uint8)
    # Maps label to continuous value.
    for i in range(num_colors):
        label[label == ids[i]] = i
        colormap[i, :] = random_color(rgb=True, maximum=255)
        if ids[i] == stuff_id:
            colormap[i, :] = np.array([0, 0, 0])
    colored_label = colormap[label]
    pan2ins = torch.from_numpy(colored_label.transpose(2, 0, 1))
    pan2ins = pan2ins.unsqueeze(dim=0).float()
    pan2ins = F.interpolate(pan2ins, size=(height, width), mode='bilinear', align_corners=False)
    pan2ins = pan2ins.squeeze(dim=0).byte()
    imgs.append(pan2ins)

    # save panoptic_pred
    # Add colormap to label.
    label_divisor = 1000
    label = panoptic_pred.copy()
    colored_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    taken_colors = set([0, 0, 0])

    def _random_color(base, max_dist=30):
        new_color = base + np.random.randint(low=-max_dist, high=max_dist + 1, size=3)
        return tuple(np.maximum(0, np.minimum(255, new_color)))

    for lab in np.unique(label):
        mask = label == lab
        base_color = color_map[lab // label_divisor]
        if tuple(base_color) not in taken_colors:
            taken_colors.add(tuple(base_color))
            color = base_color
        else:
            while True:
                color = _random_color(base_color)
                if color not in taken_colors:
                    taken_colors.add(color)
                    break
        colored_label[mask] = color

    predPanop = torch.from_numpy(colored_label.transpose(2, 0, 1))
    predPanop = predPanop.unsqueeze(dim=0).float()
    predPanop = F.interpolate(predPanop, size=(height, width), mode='bilinear', align_corners=False)
    predPanop = predPanop.squeeze(dim=0).byte()
    imgs.append(predPanop)
    #
    strTag = '{} : Panoptic-To-Semantic : Panoptic-To-Instance : Predicted-Panoptic'.format(imgid * 10 + 3)
    num_imgs_per_row = 3
    grid_image = make_grid(imgs, num_imgs_per_row, normalize=False)
    writer.add_image(strTag, grid_image, i_iter)



def draw_in_tensorboard(writer, images, i_iter, semseg_pred_source,
                        center_pred_source, offset_pred_source,
                        num_classes, label_panop_dict, dataset,
                        vis_sem_pred, vis_dep_pred, vis_cen_pred, vis_ofs_pred,
                        type=None, imgid=0):
    imgs = []
    color_map = dataset.create_label_colormap(num_classes)

    # image_source
    imgs.append(images[:3].squeeze().clone().cpu().data)

    # save gt semantic
    gt_sem = label_panop_dict['semantic'][0].cpu().numpy()
    gt_sem = label_to_color_image(gt_sem, color_map)
    imgs.append(torch.from_numpy(gt_sem.transpose(2, 0, 1)))

    # save pred semantic
    semantic_pred = torch.argmax(semseg_pred_source.detach(), dim=1)
    pred_sem = semantic_pred[0].cpu().numpy()
    pred_sem = label_to_color_image(pred_sem, color_map)
    imgs.append(torch.from_numpy(pred_sem.transpose(2, 0, 1)))

    # save gt center
    gt_ctr = label_panop_dict['center'][0].squeeze().cpu().numpy()
    gt_ctr = gt_ctr[:, :, None] * np.array([50, 0, 0]).reshape((1, 1, 3))
    gt_ctr = gt_ctr.clip(0, 255)
    imgs.append(torch.from_numpy(gt_ctr.transpose(2, 0, 1)))

    # save pred center
    if vis_cen_pred:
        pred_ctr = center_pred_source.detach().squeeze().cpu().numpy()
        pred_ctr = pred_ctr[:, :, None] * np.array([50, 0, 0]).reshape((1, 1, 3))
        pred_ctr = pred_ctr.clip(0, 255)
        imgs.append(torch.from_numpy(pred_ctr.transpose(2, 0, 1)))

    # save gt offset
    gt_off = label_panop_dict['offset'][0].permute(1, 2, 0).cpu().numpy()
    gt_off = flow_compute_color(gt_off[:, :, 1], gt_off[:, :, 0])
    imgs.append(torch.from_numpy(gt_off.transpose(2, 0, 1)))

    # save pred offset
    if vis_ofs_pred:
        pred_offset = offset_pred_source[0].detach().permute(1, 2, 0).cpu().numpy()
        pred_offset = flow_compute_color(pred_offset[:, :, 1], pred_offset[:, :, 0])
        imgs.append(torch.from_numpy(pred_offset.transpose(2, 0, 1)))

    # save semantic ignore mask
    gt_ign1 = label_panop_dict['semantic_weights'][0].cpu().numpy()
    # beacuse, semantic_weights has either values 1 or 3, it generates 50 and 150 after multipying 50
    # 1 --> light gray; 3 --> dark gray
    gt_ign1 = gt_ign1 * 50
    gt_ign1 = gt_ign1[:, :, None]
    gt_ign1 = np.tile(gt_ign1, (1, 1, 3))
    imgs.append(torch.from_numpy(gt_ign1.transpose(2, 0, 1)))

    # save center ignore mask
    gt_ign2 = label_panop_dict['center_weights'][0].cpu().numpy()
    # values are either 0 or 1; 0:Black; 1: light gray
    gt_ign2 = gt_ign2 * 50
    gt_ign2 = gt_ign2[:, :, None]
    gt_ign2 = np.tile(gt_ign2, (1, 1, 3))
    imgs.append(torch.from_numpy(gt_ign2.transpose(2, 0, 1)))

    # save offset ignore mask
    gt_ign3 = label_panop_dict['offset_weights'][0].cpu().numpy()
    # values are either 0 or 1; 0:Black; 1: light gray
    gt_ign3 = gt_ign3 * 50
    gt_ign3 = gt_ign3[:, :, None]
    gt_ign3 = np.tile(gt_ign3, (1, 1, 3))
    imgs.append(torch.from_numpy(gt_ign3.transpose(2, 0, 1)))

    strTag = '{}_{} _img-gtSem_predSem-gtCen_predCen-gtOff_predOff-semWeight_cenWeight_offWeight'.format(type, imgid)

    num_imgs_per_row = 10
    grid_image = make_grid(imgs, num_imgs_per_row, normalize=True)
    writer.add_image(strTag, grid_image, i_iter)


def per_class_iu(hist):
    np.seterr(divide='ignore', invalid='ignore')
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

# this is not in use currently
# def draw_in_tensorboard_old(writer, images, i_iter, semseg_pred_source,
#                         center_pred_source, offset_pred_source,
#                         num_classes, label_panop_dict, dataset, type=None, imgid=0):
#
#     num_imgs_per_row = 10
#
#     # image_source
#     grid_image = make_grid(images[:3].clone().cpu().data, num_imgs_per_row, normalize=True)
#     writer.add_image('1_image_{}_{}'.format(type, imgid), grid_image, i_iter)
#
#     # save gt semantic
#     gt_sem = label_panop_dict['semantic'][0].cpu().numpy()
#     gt_sem = label_to_color_image(gt_sem, dataset.create_label_colormap(num_classes))
#     grid_image = make_grid(torch.from_numpy(gt_sem.transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
#     writer.add_image('2_gt_semantic_{}_{}'.format(type, imgid), grid_image, i_iter)
#
#     # save pred semantic
#     semantic_pred = torch.argmax(semseg_pred_source.detach(), dim=1)
#     pred_sem = semantic_pred[0].cpu().numpy()
#     pred_sem = label_to_color_image(pred_sem, dataset.create_label_colormap(num_classes))
#     grid_image = make_grid(torch.from_numpy(pred_sem.transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
#     writer.add_image('3_pred_semantic_{}_{}'.format(type, imgid), grid_image, i_iter)
#
#     if type == 'source':
#         # save gt center
#         gt_ctr = label_panop_dict['center'][0].squeeze().cpu().numpy()
#         gt_ctr = gt_ctr[:, :, None] * np.array([50, 0, 0]).reshape((1, 1, 3))
#         gt_ctr = gt_ctr.clip(0, 255)
#         grid_image = make_grid(torch.from_numpy(gt_ctr.transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
#         writer.add_image('4_gt_center_{}_{}'.format(type, imgid), grid_image, i_iter)
#
#     # save pred center
#     pred_ctr = center_pred_source.detach().squeeze().cpu().numpy()
#     pred_ctr = pred_ctr[:, :, None] * np.array([50, 0, 0]).reshape((1, 1, 3))
#     pred_ctr = pred_ctr.clip(0, 255)
#     grid_image = make_grid(torch.from_numpy(pred_ctr.transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
#     writer.add_image('5_pred_center_{}_{}'.format(type, imgid), grid_image, i_iter)
#
#     if type == 'source':
#         # save gt offset
#         gt_off = label_panop_dict['offset'][0].permute(1, 2, 0).cpu().numpy()
#         gt_off = flow_compute_color(gt_off[:, :, 1], gt_off[:, :, 0])
#         grid_image = make_grid(torch.from_numpy(gt_off.transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
#         writer.add_image('6_gt_offset_{}_{}'.format(type, imgid), grid_image, i_iter)
#
#     # save pred offset
#     pred_offset = offset_pred_source[0].detach().permute(1, 2, 0).cpu().numpy()
#     pred_offset = flow_compute_color(pred_offset[:, :, 1], pred_offset[:, :, 0])
#     grid_image = make_grid(torch.from_numpy(pred_offset.transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
#     writer.add_image('7_pred_offset_{}_{}'.format(type, imgid), grid_image, i_iter)
#
#     # for tensorboard - 0 and 255 both appears black, so I need to chose values like 50 and 150, show that it shows
#     # So, in tensor board both 50 and 150 pixel values show GRAY volour
#     # 50 is lighter gray; and 150 is darker gray
#
#     if type == 'source':
#         # save semantic ignore mask
#         gt_ign1 = label_panop_dict['semantic_weights'][0].cpu().numpy()
#         # beacuse, semantic_weights has either values 1 or 3, it generates 50 and 150 after multipying 50
#         # 1 --> light gray; 3 --> dark gray
#         gt_ign1 = gt_ign1 * 50
#         gt_ign1 = gt_ign1[:, :, None]
#         gt_ign1 = np.tile(gt_ign1, (1, 1, 3))
#         grid_image = make_grid(torch.from_numpy(gt_ign1.transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
#         writer.add_image('8_semantic_ignore_mask_{}_{}'.format(type, imgid), grid_image, i_iter)
#
#         # save center ignore mask
#         gt_ign2 = label_panop_dict['center_weights'][0].cpu().numpy()
#         # values are either 0 or 1; 0:Black; 1: light gray
#         gt_ign2 = gt_ign2 * 50
#         gt_ign2 = gt_ign2[:, :, None]
#         gt_ign2 = np.tile(gt_ign2, (1, 1, 3))
#         grid_image = make_grid(torch.from_numpy(gt_ign2.transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
#         writer.add_image('9_center_ignore_mask_{}_{}'.format(type, imgid), grid_image, i_iter)
#
#         # save offset ignore mask
#         gt_ign3 = label_panop_dict['offset_weights'][0].cpu().numpy()
#         # values are either 0 or 1; 0:Black; 1: light gray
#         gt_ign3 = gt_ign3 * 50
#         gt_ign3 = gt_ign3[:, :, None]
#         gt_ign3 = np.tile(gt_ign3, (1, 1, 3))
#         grid_image = make_grid(torch.from_numpy(gt_ign3.transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
#         writer.add_image('10_offset_ignore_mask_{}_{}'.format(type, imgid), grid_image, i_iter)

    # -----------------------------------
    # semseg_pred_source
    # softmax = F.softmax(semseg_pred_source, dim=1).cpu().data[0].numpy().transpose(1, 2, 0)
    # mask = colorize_mask(num_classes, np.asarray(np.argmax(softmax, axis=2), dtype=np.uint8)).convert("RGB")
    # grid_image = make_grid(torch.from_numpy(np.array(mask).transpose(2, 0, 1)), num_imgs_per_row, normalize=False, range=(0, 255))
    # writer.add_image('semseg_pred_source', grid_image, i_iter)





