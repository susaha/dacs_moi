from ctrl.utils.panoptic_deeplab.utils import AverageMeter
from ctrl.eval_panop import SemanticEvaluator
from ctrl.eval_panop import CityscapesInstanceEvaluator
from ctrl.eval_panop import CityscapesPanopticEvaluator
from ctrl.model_panop.post_processing import get_semantic_segmentation, get_panoptic_segmentation
import cv2
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from fvcore.common.file_io import PathManager
from ctrl.model_panop.post_processing import get_cityscapes_instance_format
from ctrl.utils.panoptic_deeplab import save_debug_images
from ctrl.utils.panoptic_deeplab import save_annotation, save_instance_annotation, save_panoptic_annotation, save_heatmap_image, save_offset_image_v2
from ctrl.utils.panoptic_deeplab.logger import setup_logger
import logging
import pprint
from random import randint
import random
from ctrl.utils.train_utils import draw_in_tensorboard_val
from ctrl.utils.panoptic_deeplab.utils import get_loss_info_str, get_module
from ctrl.utils.panoptic_deeplab import comm
from ctrl.utils.panoptic_deeplab.utils import get_module
import torch.nn as nn



def eval_panoptic(config, model, eval_folder, writer, i_iter, logger, device, data_loader, img_ids=None):


    if 'Cityscapes' in config.TARGET:
        panoptic_josn_file = 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}.json'
        panoptic_json_folder = 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}'
        stuff_area = config.POST_PROCESSING.STUFF_AREA # 2048
    elif 'Mapillary' in config.TARGET:
        panoptic_josn_file = '{}/val_panoptic_1024x768.json'.format(config.DATA_DIRECTORY_TARGET)
        panoptic_json_folder = '{}/val_labels'.format(config.DATA_DIRECTORY_TARGET)
        stuff_area = 4096 # this is as per panoptic deeplab paper: --> ' The thresholds on Cityscapes, Mapillary Vistas, and COCO are 2048, 4096, and 4096, respectively.'

    data_time = AverageMeter()
    net_time = AverageMeter()
    post_time = AverageMeter()
    timing_warmup_iter = 10

    semantic_metric = SemanticEvaluator(
        num_classes=config.NUM_CLASSES,
        ignore_label=config.PANOPTIC_TARGET_GENERATOR.IGNORE_LABEL,
        output_dir=eval_folder['semantic'],
        train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(config.NUM_CLASSES)
    )

    instance_metric = None
    panoptic_metric = None

    if config.TEST.EVAL_INSTANCE:
        if 'Cityscapes' in config.TARGET or 'Mapillary' in config.TARGET:
            if 'Cityscapes' in config.TARGET:
                gt_dir = os.path.join(data_loader.dataset.root, 'gtFineV2', data_loader.dataset.set)
                logger.info('ctrl/evaluate_panop_jan19_2022.py - gt_dir: ')
                logger.info(gt_dir)
            elif 'Mapillary' in config.TARGET:
                gt_dir = '{}/{}_labels'.format(config.DATA_DIRECTORY_TARGET, 'val')

            instance_metric = CityscapesInstanceEvaluator(
                output_dir=eval_folder['instance'],
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(config.NUM_CLASSES),
                gt_dir=gt_dir,
                num_classes=config.NUM_CLASSES,
                DEBUG=config.DEBUG,
                num_samples=config.NUM_VAL_SAMPLES_DURING_DEBUG,
                dataset_name=config.TARGET,
                rgb2id=data_loader.dataset.rgb2id,
                input_image_size=data_loader.dataset.image_size,
                mapillary_dataloading_style=config.MAPILLARY_DATA_LOADING_STYLE,
            )

    if config.TEST.EVAL_PANOPTIC:
        if 'Cityscapes' in config.TARGET or 'Mapillary' in config.TARGET:
            if 'Cityscapes' in config.TARGET:
                gt_dir = data_loader.dataset.root
            elif 'Mapillary' in config.TARGET:
                gt_dir = None
            panoptic_metric = CityscapesPanopticEvaluator(
                output_dir=eval_folder['panoptic'],
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(config.NUM_CLASSES),
                label_divisor=data_loader.dataset.label_divisor,
                void_label=data_loader.dataset.label_divisor * config.PANOPTIC_TARGET_GENERATOR.IGNORE_LABEL,
                gt_dir=gt_dir,
                split=data_loader.dataset.set,
                num_classes=config.NUM_CLASSES,
                panoptic_josn_file=panoptic_josn_file,
                panoptic_json_folder=panoptic_json_folder,
                DEBUG=config.DEBUG,
                target_dataset_name=config.TARGET,
                input_image_size=data_loader.dataset.image_size,
                mapillary_dataloading_style=config.MAPILLARY_DATA_LOADING_STYLE,

            )

    if 'Cityscapes' in config.TARGET:
        image_filename_list = [os.path.splitext(os.path.basename(ann[2]))[0] for ann in data_loader.dataset.files]
        fixed_test_size = True
    elif 'Mapillary' in config.TARGET:
        image_filename_list = [os.path.splitext(os.path.basename(ann[0]))[0] for ann in data_loader.dataset.files]
        fixed_test_size = False


    # Debug output.
    if config.DEBUG:
        debug_out_dir = eval_folder['debug_test']

    # Evaluation loop.
    num_samples = len(data_loader)
    tbc = 0
    try:
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if config.DEBUG and i >= 20: # config.NUM_VAL_SAMPLES_DURING_DEBUG:
                    break
                image, img_name = batch
                if i == timing_warmup_iter:
                    data_time.reset()
                    net_time.reset()
                    post_time.reset()
                # data
                start_time = time.time()

                image = image.to(device)
                torch.cuda.synchronize(device)
                data_time.update(time.time() - start_time)
                start_time = time.time()
                out_dict = model(image)

                upsample_dim = (config.TEST.OUTPUT_SIZE_TARGET[1], config.TEST.OUTPUT_SIZE_TARGET[0])  # H x W

                out_dict = get_module(model, False).upsample_predictions(out_dict, upsample_dim)
                torch.cuda.synchronize(device)
                net_time.update(time.time() - start_time)
                start_time = time.time()
                semantic_pred = get_semantic_segmentation(out_dict['semantic'])
                if 'foreground' in out_dict:
                    foreground_pred = get_semantic_segmentation(out_dict['foreground'])
                else:
                    foreground_pred = None
                if config.TEST.EVAL_INSTANCE or config.TEST.EVAL_PANOPTIC:
                    panoptic_pred, center_pred = get_panoptic_segmentation(
                        semantic_pred,
                        out_dict['center'],
                        out_dict['offset'],
                        thing_list=data_loader.dataset.cityscapes_thing_list,
                        label_divisor=1000, # data_loader.dataset.label_divisor,
                        stuff_area=4096, # stuff_area,
                        void_label=(1000 * 255),
                        threshold=config.POST_PROCESSING.CENTER_THRESHOLD,
                        nms_kernel=config.POST_PROCESSING.NMS_KERNEL,
                        top_k=config.POST_PROCESSING.TOP_K_INSTANCE,
                        foreground_mask=foreground_pred)
                else:
                    panoptic_pred = None

                torch.cuda.synchronize(device)
                post_time.update(time.time() - start_time)
                if i % config.TEST.DISPLAY_LOG_EVERY == 0 or config.DEBUG:
                    logger.info('[{}/{}]\t'
                                'Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                                'Network Time: {net_time.val:.3f}s ({net_time.avg:.3f}s)\t'
                                'Post-processing Time: {post_time.val:.3f}s ({post_time.avg:.3f}s)\t'.format(
                                 i, num_samples, data_time=data_time, net_time=net_time, post_time=post_time))

                semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
                if panoptic_pred is not None:
                    panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()
                if foreground_pred is not None:
                    foreground_pred = foreground_pred.squeeze(0).cpu().numpy()

                im4Vis = image.detach().clone()
                interp_im4Vis = nn.Upsample(size=(config.TEST.OUTPUT_SIZE_TARGET[1],
                                                  config.TEST.OUTPUT_SIZE_TARGET[0]),
                                            mode="bilinear", align_corners=True, )
                im4Vis = interp_im4Vis(im4Vis)
                im4Vis = im4Vis.squeeze().cpu().numpy()
                im4Vis = im4Vis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
                im4Vis += config.TRAIN.IMG_MEAN
                im4Vis = im4Vis[:, :, ::-1]  # BGR --> RGB

                str2 = img_name[0].split('.')[0]
                str3 = str2.split('_')
                vid_id = str3[1]

                # dump instance segmentation visual images
                out_folder_name = os.path.join(eval_folder['ins_vis'], vid_id)
                if not os.path.exists(out_folder_name):
                    os.makedirs(out_folder_name)
                    print('folder  created {}'.format(out_folder_name))
                out_file_name = str3[2] + '_' + str3[3]
                ins_id = panoptic_pred % 1000
                sem_id = np.around(panoptic_pred / 1000).astype(int)
                no_person_mask = np.logical_and(sem_id != 10, sem_id != 11)
                pan_to_ins = panoptic_pred.copy()
                pan_to_ins[ins_id == 0] = 0
                pan_to_ins[no_person_mask] = 0
                save_instance_annotation(pan_to_ins,  out_folder_name,  out_file_name, image=im4Vis,  blend_ratio=0.7,   DEBUG=False)

                # dump semantic segmentation visual images
                out_folder_name = os.path.join(eval_folder['sem_vis'], vid_id)
                if not os.path.exists(out_folder_name):
                    os.makedirs(out_folder_name)
                    print('folder  created {}'.format(out_folder_name))
                save_annotation(semantic_pred, out_folder_name,  out_file_name,  add_colormap=True,
                                colormap=data_loader.dataset.create_label_colormap(2), image=im4Vis, blend_ratio=0.7, DEBUG=False)

                # dump center heatmap
                center_hmp = out_dict['center']
                center_hmp = F.interpolate(center_hmp, size=upsample_dim, mode='bilinear', align_corners=False)
                center_hmp = center_hmp.squeeze(1).squeeze(0).cpu().numpy()
                center_hmp[np.logical_and(semantic_pred!=10, semantic_pred!=11)] = 0
                out_folder_name = os.path.join(eval_folder['ins_vis'] + '_center', vid_id)
                if not os.path.exists(out_folder_name):
                    os.makedirs(out_folder_name)
                    print('folder  created {}'.format(out_folder_name))
                save_heatmap_image(im4Vis, center_hmp, out_folder_name, out_file_name, ratio=0.7, DEBUG=False)

                # dump offset heatmap
                out_folder_name = os.path.join(eval_folder['ins_vis'] + '_offset', vid_id)
                if not os.path.exists(out_folder_name):
                    os.makedirs(out_folder_name)
                    print('folder  created {}'.format(out_folder_name))
                offset_hmp = out_dict['offset']
                offset_hmp = F.interpolate(offset_hmp, size=upsample_dim, mode='bilinear', align_corners=False)
                offset_hmp = offset_hmp.squeeze(1).squeeze(0).cpu().numpy().transpose([1,2,0])
                offset_hmp[np.logical_and(semantic_pred != 10, semantic_pred != 11), :] = 0
                save_offset_image_v2(im4Vis, offset_hmp, out_folder_name, out_file_name, ratio=0.7, DEBUG=False)


                # dump panoptic segmentation visual images
                # save_panoptic_annotation(panoptic_pred,
                #                             eval_folder['pan_vis'],
                #                             vis_fname,
                #                             label_divisor=data_loader.dataset.label_divisor,
                #                             colormap=data_loader.dataset.create_label_colormap(config.NUM_CLASSES),
                #                             image=im4Vis,
                #                             blend_ratio=0.7)


    except Exception:
        logger.exception("Exception during testing:")
        raise
    finally:

        logger.info("Inference finished.")
