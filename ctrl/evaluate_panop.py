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
from ctrl.utils.panoptic_deeplab import save_annotation, save_instance_annotation, save_panoptic_annotation
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

    '''
        eval_folder['semantic'] = 'outpath/eval/semanitc'
        eval_folder['instance'] = 'outpath/eval/instance'
        eval_folder['panoptic'] = 'outpath/eval/panoptic'
        eval_folder['debug_test'] = 'outpath/eval/debug_test'
        eval_folder['logger'] = 'outpath/eval/logger'
    '''

    # setup logger
    # logger = logging.getLogger('segmentation_test')
    # if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
    #     setup_logger(output=eval_folder['logger_eval'], name='segmentation_test')

    # import pickle
    # pckl_file = os.path.join(eval_folder['debug_test'], 'eval_panop_debug.pkl')
    # outfile = open(pckl_file, 'wb')
    # logger.info(pckl_file)


    # select the correct json file and folder (which contains the PNG panoptic gt label files) for evaluation
    # NOTE: for training the file and folder names end with _trainId
    # if config.NUM_CLASSES == 16:
    panoptic_josn_file = 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}.json'
    panoptic_json_folder = 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}'
    # elif config.NUM_CLASSES == 19:
    #     panoptic_josn_file = 'cityscapes_panoptic_{}.json'
    #     panoptic_json_folder = 'cityscapes_panoptic_{}'

    # THIS GIVES ERROR
    # if config.NUM_CLASSES == 16:
    #     panoptic_josn_file = 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}_trainId.json'
    #     panoptic_json_folder = 'cityscapes_panoptic_synthia_to_cityscapes_16cls_{}_trainId'
    # elif config.NUM_CLASSES == 19:
    #     panoptic_josn_file = 'cityscapes_panoptic_{}_trainId.json'
    #     panoptic_json_folder = 'cityscapes_panoptic_{}_trainId'


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

    # config.TEST.EVAL_INSTANCE=False

    if config.TEST.EVAL_INSTANCE:
        if 'Cityscapes' in config.TARGET:
            instance_metric = CityscapesInstanceEvaluator(
                output_dir=eval_folder['instance'],
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(config.NUM_CLASSES),
                gt_dir=os.path.join(data_loader.dataset.root, 'gtFine', data_loader.dataset.set),
                num_classes=config.NUM_CLASSES,
                DEBUG=config.DEBUG,
                num_samples=config.NUM_VAL_SAMPLES_DURING_DEBUG,
            )
        if 'Mapillary' in config.TARGET:
            raise ValueError('MapillaryInstanceEvaluator is not defined !!')

    if config.TEST.EVAL_PANOPTIC:
        if 'Cityscapes' in config.TARGET:
            panoptic_metric = CityscapesPanopticEvaluator(
                output_dir=eval_folder['panoptic'],
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(config.NUM_CLASSES),
                label_divisor=data_loader.dataset.label_divisor,
                void_label=data_loader.dataset.label_divisor * config.PANOPTIC_TARGET_GENERATOR.IGNORE_LABEL,
                gt_dir=data_loader.dataset.root,
                split=data_loader.dataset.set,
                num_classes=config.NUM_CLASSES,
                panoptic_josn_file=panoptic_josn_file,
                panoptic_json_folder=panoptic_json_folder,
                DEBUG=config.DEBUG
            )
        if 'Mapillary' in config.TARGET:
            raise ValueError('MapillaryPanopticEvaluator is not defined !!')

    # img_file, label_file, panop_label_file, disp_file, calib_file, segment_info, name = self.files[index]
    image_filename_list = [os.path.splitext(os.path.basename(ann[2]))[0] for ann in data_loader.dataset.files]


    # Debug output.
    if config.DEBUG:
        debug_out_dir = eval_folder['debug_test']
        # PathManager.mkdirs(debug_out_dir)

    # Evaluation loop.
    num_samples = len(data_loader)
    tbc = 0
    try:
        # get_module(model, config.DISTRIBUTED).eval()
        model.eval()

        with torch.no_grad():

            # return image.copy(), label_panop_dict, depth_labels.copy(), np.array(image.shape), name
            for i, batch in enumerate(data_loader):

                if config.DEBUG and i >= config.NUM_VAL_SAMPLES_DURING_DEBUG:
                    break

                image, data, image_size, img_name = batch
                # print(image.shape)
                # image, semseg_label, data, image_size, img_name = batch
                # image.copy(), semseg_label, panop_lbl_dict, np.array(image.shape), name
                if i == timing_warmup_iter:
                    data_time.reset()
                    net_time.reset()
                    post_time.reset()

                # data
                start_time = time.time()
                # if not config.DISTRIBUTED:
                for key in data.keys():
                    try:
                        data[key] = data[key].to(device)
                        # print('GT labels: data[{}]={}'.format(key, data[key].shape))
                    except:
                        pass
                image = image.to(device)

                # image = data.pop('image')
                torch.cuda.synchronize(device)
                data_time.update(time.time() - start_time)

                start_time = time.time()
                # if config.TEST.TEST_TIME_AUGMENTATION:
                #     raise ValueError('I need to implement this part (refer to /home/suman/apps/code/CVPR2022/panoptic-deeplab/tools/test_net_single_core.py) line-no: 215')
                # else:
                # forward pass
                out_dict = model(image)

                # resizing the prediction maps back to original input image size
                # when augmentation is activated, then first the predictions
                # are resized to the crop_size which is 1 pixel larger than the
                # the original input size both in width and height
                # if config.PANOPTIC_DEEPLAB_DATA_AUG_ALL:
                #     input_size = config.DATASET.CROP_SIZE
                # else:
                input_size = config.TEST.OUTPUT_SIZE_TARGET
                input_size = (input_size[1], input_size[0])  # H x W
                out_dict = get_module(model, False).upsample_predictions(out_dict, input_size)

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
                        label_divisor=data_loader.dataset.label_divisor,
                        stuff_area=config.POST_PROCESSING.STUFF_AREA,
                        void_label=(
                                data_loader.dataset.label_divisor *
                                data_loader.dataset.ignore_label),
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

                # Crop padded regions.
                # if config.PANOPTIC_DEEPLAB_DATA_AUG_ALL:
                #     image_size = [config.DATASET.CROP_SIZE[1]-1, config.DATASET.CROP_SIZE[0]-1]
                #     semantic_pred = semantic_pred[:image_size[0], :image_size[1]]
                #     if panoptic_pred is not None:
                #         panoptic_pred = panoptic_pred[:image_size[0], :image_size[1]]
                #     if foreground_pred is not None:
                #         foreground_pred = foreground_pred[:image_size[0], :image_size[1]]
                #
                #     # resize the prediction to the original image size
                #     H1 = config.TEST.OUTPUT_SIZE_TARGET[1]
                #     H2 = config.DATASET.CROP_SIZE[1]-1
                #     W1 = config.TEST.OUTPUT_SIZE_TARGET[0]
                #     W2 = config.DATASET.CROP_SIZE[0] - 1
                #     if H1 != H2 or W1 != W2:
                #         if semantic_pred is not None:
                #             semantic_pred = cv2.resize(semantic_pred.astype(np.float),
                #                                        (config.TEST.OUTPUT_SIZE_TARGET[0], config.TEST.OUTPUT_SIZE_TARGET[1]),
                #                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)
                #         if panoptic_pred is not None:
                #             panoptic_pred = cv2.resize(panoptic_pred.astype(np.float),
                #                                        (config.TEST.OUTPUT_SIZE_TARGET[0], config.TEST.OUTPUT_SIZE_TARGET[1]),
                #                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)
                #         if foreground_pred is not None:
                #             foreground_pred = cv2.resize(foreground_pred.astype(np.float),
                #                                          (config.TEST.OUTPUT_SIZE_TARGET[0], config.TEST.OUTPUT_SIZE_TARGET[1]),
                #                                          interpolation=cv2.INTER_NEAREST).astype(np.int32)

                # Evaluates semantic segmentation.
                semantic_metric.update(semantic_pred, data['semantic'].squeeze(0).cpu().numpy(), image_filename_list[i]) # TODO: this worked so far
                # semantic_metric.update(semantic_pred, data['raw_label'].squeeze(0).cpu().numpy(), image_filename_list[i]) # TODO: for a sanity check try this  once

                # Optional: evaluates instance segmentation.
                if instance_metric is not None:
                    raw_semantic = F.softmax(out_dict['semantic'], dim=1)
                    raw_semantic = F.interpolate(raw_semantic, size=(config.TEST.OUTPUT_SIZE_TARGET[1], config.TEST.OUTPUT_SIZE_TARGET[0]),
                                                 mode='bilinear', align_corners=False)  # Consistent with OpenCV.
                    center_hmp = out_dict['center']
                    center_hmp = F.interpolate(center_hmp, size=(config.TEST.OUTPUT_SIZE_TARGET[1], config.TEST.OUTPUT_SIZE_TARGET[0]),
                                               mode='bilinear', align_corners=False)  # Consistent with OpenCV.

                    raw_semantic = raw_semantic.squeeze(0).cpu().numpy()
                    center_hmp = center_hmp.squeeze(1).squeeze(0).cpu().numpy()

                    instances = get_cityscapes_instance_format(panoptic_pred, raw_semantic, center_hmp, label_divisor=data_loader.dataset.label_divisor,
                                                               score_type=config.TEST.INSTANCE_SCORE_TYPE)

                    instance_metric.update(instances, image_filename_list[i])

                # Optional: evaluates panoptic segmentation.
                if panoptic_metric is not None:
                    image_id = '_'.join(image_filename_list[i].split('_')[:3])
                    panoptic_metric.update(panoptic_pred, image_filename=image_filename_list[i], image_id=image_id)

                # if config.TEST.DEBUG: # False
                #     # Raw outputs
                #     save_debug_images(
                #         dataset=data_loader.dataset,
                #         batch_images=image,
                #         batch_targets=data,
                #         batch_outputs=out_dict,
                #         out_dir=debug_out_dir,
                #         iteration=i,
                #         target_keys=tuple(config.PANOPTIC_TARGET_GENERATOR.TARGET_KEYS),
                #         output_keys=tuple(config.PANOPTIC_TARGET_GENERATOR.OUTPUT_KEYS),
                #         is_train=False,
                #     )

                if config.DUMP_PANOPTIC_VISUAL_IMGS:
                    im4Vis = image.detach().clone()
                    interp_im4Vis = nn.Upsample(size=(config.TEST.OUTPUT_SIZE_TARGET[1],
                                                      config.TEST.OUTPUT_SIZE_TARGET[0]),
                                                mode="bilinear", align_corners=True, )  #  size=(1024, 2048)
                    im4Vis = interp_im4Vis(im4Vis)
                    im4Vis = im4Vis.squeeze().cpu().numpy()
                    im4Vis = im4Vis.transpose((1, 2, 0))  # C x H x W  --> H x W x C
                    im4Vis += config.TRAIN.IMG_MEAN
                    im4Vis = im4Vis[:, :, ::-1]  # BGR --> RGB
                    # dump panoptic segmentation visual images
                    vis_fname = img_name[0].split('.')[0]

                    save_panoptic_annotation(panoptic_pred,
                                                eval_folder['pan_vis'],
                                                vis_fname,
                                                label_divisor=data_loader.dataset.label_divisor,
                                                colormap=data_loader.dataset.create_label_colormap(config.NUM_CLASSES),
                                                image=im4Vis,
                                                blend_ratio=0.7)

                    # dump instance segmentation visual images
                    ins_id = panoptic_pred % data_loader.dataset.label_divisor
                    pan_to_ins = panoptic_pred.copy()
                    pan_to_ins[ins_id == 0] = 0

                    save_instance_annotation(pan_to_ins,
                                             eval_folder['ins_vis'],
                                             vis_fname,
                                             image=im4Vis,
                                             blend_ratio=0.7)

                    # dump semantic segmentation visual images
                    save_annotation(semantic_pred,
                                    eval_folder['sem_vis'],
                                    vis_fname,
                                    add_colormap=True,
                                    colormap=data_loader.dataset.create_label_colormap(config.NUM_CLASSES),
                                    image=im4Vis,
                                    blend_ratio=0.7)
                    # print('vis_fname: {}'.format(vis_fname))


                # Processed outputs
                # if False: # panoptic_pred is not None and i in img_ids:
                #     tbc += 1
                #     debug_out_dir = eval_folder['debug_test']
                #     save_annotation(semantic_pred, debug_out_dir, 'semantic_pred_%d' % i, add_colormap=True, colormap=data_loader.dataset.create_label_colormap(config.NUM_CLASSES))
                #
                #     pan_to_sem = panoptic_pred // data_loader.dataset.label_divisor
                #     save_annotation(pan_to_sem, debug_out_dir, 'pan_to_sem_pred_%d' % i, add_colormap=True, colormap=data_loader.dataset.create_label_colormap(config.NUM_CLASSES))
                #
                #     ins_id = panoptic_pred % data_loader.dataset.label_divisor
                #     pan_to_ins = panoptic_pred.copy()
                #     pan_to_ins[ins_id == 0] = 0
                #     save_instance_annotation(pan_to_ins, debug_out_dir, 'pan_to_ins_pred_%d' % i)
                #
                #     save_panoptic_annotation(panoptic_pred, debug_out_dir, 'panoptic_pred_%d' % i, label_divisor=data_loader.dataset.label_divisor,
                #                              colormap=data_loader.dataset.create_label_colormap(config.NUM_CLASSES))
                #     # print('instance ids in panoptic array:')
                #     # print(np.unique(panoptic_pred))
                #     # def save_panoptic_annotation(label, save_dir, filename, label_divisor,  colormap=None, image=None):
                #
                #
                #     draw_in_tensorboard_val(writer, image, i_iter, semantic_pred, pan_to_sem,  pan_to_ins, panoptic_pred, config.NUM_CLASSES, data, data_loader.dataset, tbc)
                #     # draw_in_tensorboard_val(writer, image, i_iter, semantic_pred.detach(), panoptic_pred.detach(), config.NUM_CLASSES, data, data_loader.dataset, 'target', i)
                #
                #     logger.info('predictions and gt labels for cityscapes eval image id {} are displayed in tensorbaord ... '.format(i))
                #     logger.info('proccessed outputs for imageid: {} is saved to {} ... '.format(i, debug_out_dir))


    except Exception:
        logger.exception("Exception during testing:")
        raise
    finally:

        # metrics = {}
        # metrics['semanitc'] = semantic_metric
        # metrics['instance'] = instance_metric
        # metrics['panoptic'] = panoptic_metric
        # pickle.dump(metrics, outfile)
        # outfile.close()
        # print()

        logger.info("Inference finished.")
        semantic_results = semantic_metric.evaluate()
        logger.info(semantic_results)
        if instance_metric is not None:
            instance_results = instance_metric.evaluate()
            logger.info(instance_results)
        if panoptic_metric is not None:
            panoptic_results = panoptic_metric.evaluate()
            logger.info(panoptic_results)


    # ----------- set the model back to train() mode -------------
    # get_module(model, config.DISTRIBUTED).set_image_pooling(None)
    # get_module(model, config.DISTRIBUTED).train()
    # -----------
