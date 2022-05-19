from torch.utils.tensorboard import SummaryWriter
import os
from ctrl.utils.train_utils import adjust_learning_rate, adjust_learning_rate_disc, print_losses,\
    get_checkpoint_path, save_checkpoint, log_losses_tensorboard_cvpr2021, log_losses_tensorboard_cvpr2022, draw_in_tensorboard
import torch
from ctrl.utils.panoptic_deeplab import AverageMeter
from ctrl.utils.panoptic_deeplab import comm
import time
from collections import OrderedDict
from ctrl.utils.panoptic_deeplab.utils import get_loss_info_str, get_module
import logging
from ctrl.eval_semantics import eval_model
from ctrl.model_panop.inst_seg import InstSeg
from ctrl.train_panop_utils import disc_fwd_pass, create_panop_eval_dirs
import random
from ctrl.train_panop_utils import set_nets_mode
from ctrl.evaluate_panop import eval_panoptic
import numpy as np
from ctrl.dacs_old.utils.train_dacs import train_dacs_one_iter
from ctrl.dacs_old.utils.train_uda_scripts import update_ema_variables
from time import perf_counter


def train_model_dacs(cfg, model, ema_model, discriminator, discriminator2nd, resume_iteration,
                criterion_dict, optimizer, optimizer_disc, optimizer_disc2nd,
                source_train_loader, target_train_loader, target_val_loader,
                source_train_nsamp, target_train_nsamp, target_test_nsamp,
                lr_scheduler, best_param_group_id, lr_scheduler_disc, DEVICE):

    logger = logging.getLogger(__name__)
    panop_eval_folder_dict = None
    panop_eval_writer = None
    if cfg.ACTIVATE_PANOPTIC_EVAL:
        # panop_eval_root_folder creates a unique eval root folder using the current time stamp
        panop_eval_folder_dict, panop_eval_root_folder = create_panop_eval_dirs(cfg.TRAIN.SNAPSHOT_DIR, logger)
        panop_eval_writer = SummaryWriter(log_dir=panop_eval_folder_dict['tensorboard'])

    criterion_center = None
    criterion_offset = None
    criterion_depth = None
    criterion_disc = None
    criterion_disc2nd = None
    dacs_unlabeled_loss_semantic = None
    criterion_semseg = criterion_dict['semseg']
    if cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
        criterion_center = criterion_dict['center']
        criterion_offset = criterion_dict['offset']
    if cfg.TRAIN.TRAIN_DEPTH_BRANCH:
        criterion_depth = criterion_dict['depth']
    if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
        criterion_disc = criterion_dict['disc']
    if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR_2ND:
        criterion_disc2nd = criterion_dict['disc2nd']
    if cfg.TRAIN.TRAIN_WITH_DACS:
        dacs_unlabeled_loss_semantic = criterion_dict['dacs_unlabeled_loss_semantic']

    inst_seg_source = None
    inst_seg_target = None
    if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
        if cfg.APPROACH_TYPE == 'DANDA':
            thing_list_source = source_train_loader.dataset.synthia_thing_list
            inst_seg_source = InstSeg(cfg, thing_list_source)
            thing_list_target = target_train_loader.dataset.cityscapes_thing_list
            inst_seg_target = InstSeg(cfg, thing_list_target)

    if cfg.PANOPTIC_DEEPLAB_DATA_AUG_ALL:
        input_size_source = cfg.DATASET.CROP_SIZE
    elif cfg.PANOPTIC_DEEPLAB_DATA_AUG_RANDOM_CROP or cfg.DACS_RANDOM_CROP:
        input_size_source = (cfg.DATASET.RANDOM_CROP_DIM, cfg.DATASET.RANDOM_CROP_DIM)
    else:
        input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE

    input_size_target = None
    if not cfg.TRAIN_ONLY_SOURCE:
        if cfg.PANOPTIC_DEEPLAB_DATA_AUG_RANDOM_CROP or cfg.DACS_RANDOM_CROP:
            input_size_target = (cfg.DATASET.RANDOM_CROP_DIM, cfg.DATASET.RANDOM_CROP_DIM)
        else:
            input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET

    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    writer = None
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    get_module(model, cfg.DISTRIBUTED).train()
    get_module(ema_model, cfg.DISTRIBUTED).train()


    if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
        get_module(discriminator, cfg.DISTRIBUTED).train()
    if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR_2ND:
        get_module(discriminator2nd, cfg.DISTRIBUTED).train()

    # interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode="bilinear", align_corners=True, )
    input_shape_source = (input_size_source[1], input_size_source[0])  # H x W

    input_shape_target = None
    source_label = None
    target_label = None
    if not cfg.TRAIN_ONLY_SOURCE:
        input_shape_target = (input_size_target[1], input_size_target[0])
        source_label = 0
        target_label = 1

    source_train_loader_iter = enumerate(source_train_loader)
    target_train_loader_iter = None
    if not cfg.TRAIN_ONLY_SOURCE:
        target_train_loader_iter = enumerate(target_train_loader)

    local_iter = 0
    current_epoch = 0
    if not cfg.TRAIN_ONLY_SOURCE:
        if source_train_nsamp > target_train_nsamp:
            iterInOneEpoch = int(source_train_nsamp / cfg.TRAIN.IMS_PER_BATCH)
        else:
            iterInOneEpoch = int(target_train_nsamp / cfg.TRAIN.IMS_PER_BATCH)
    else:
        iterInOneEpoch = int(source_train_nsamp / cfg.TRAIN.IMS_PER_BATCH)

    num_train_epochs = int(cfg.TRAIN.MAX_ITER / iterInOneEpoch)
    logger.info('num iterations in one epoch: {}'.format(iterInOneEpoch))
    logger.info('total epoch to train: {}'.format(num_train_epochs))

    logger.info('*** cfg in yaml format ***')
    cfg_print = dict(cfg)
    for k, v in cfg_print.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        logger.info('cfg.{}.{}.{}: {}'.format(k, k1, k2, v2))
                else:
                    logger.info('cfg.{}.{}: {}'.format(k, k1, v1))
        else:
            logger.info('cfg.{}: {}'.format(k, v))


    data_time = AverageMeter()
    batch_time = AverageMeter()
    # loss_meter = AverageMeter()
    loss_meter_dict = OrderedDict()
    loss_meter_dict['loss'] = AverageMeter()
    loss_meter_dict['loss_semantic'] = AverageMeter()
    if cfg.TRAIN.TRAIN_INSTANCE_BRANCH:
        loss_meter_dict['loss_center'] = AverageMeter()
        loss_meter_dict['loss_offset'] = AverageMeter()
    if cfg.TRAIN.TRAIN_DEPTH_BRANCH:
        loss_meter_dict['loss_depth'] = AverageMeter()
    if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
        loss_meter_dict['loss_adv_gen'] = AverageMeter()
        loss_meter_dict['loss_disc'] = AverageMeter()
    if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR_2ND:
        loss_meter_dict['loss_adv_gen2nd'] = AverageMeter()
        loss_meter_dict['loss_disc2nd'] = AverageMeter()
    if cfg.TRAIN.TRAIN_WITH_DACS:
        loss_meter_dict['loss_dacs_unlabeled_semantic'] = AverageMeter()

    if resume_iteration:
        start_iter = resume_iteration + 1
    else:
        start_iter = 0
    max_iter = cfg.TRAIN.MAX_ITER
    best_miou = -1
    best_checkpoint_path = ''

    logger.info('')
    logger.info('*** TRAINING STARTS HERE ***')
    logger.info('')

    loss_adv_gen2nd = 0.0
    loss_d2nd = 0

    # TODO: comment it later
    perf_time = {}
    perf_time['src_data'] = []
    perf_time['forward_pass_data'] = []
    perf_time['sup_loss_data'] = []
    perf_time['tar_data'] = []
    perf_time['unlabeled_train_time'] = []
    perf_time['backward_pass_time'] = []
    perf_time['mean_teacher_update_time'] = []
    # max_iter = 100

    # train loop
    try:
        for i_iter in range(start_iter, max_iter):

            # epoch count
            if (i_iter+1) % iterInOneEpoch == 0:
                current_epoch+=1
            local_iter += 1

            optimizer.zero_grad()
            if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
                optimizer_disc.zero_grad()
                for param in discriminator.parameters():
                    param.requires_grad = False
            if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR_2ND:
                optimizer_disc2nd.zero_grad()
                for param in discriminator2nd.parameters():
                    param.requires_grad = False

            # start_time = time.time()
            start_time = perf_counter()
            # supervised training on source
            try:
                _, batch = source_train_loader_iter.__next__()
            except StopIteration:
                source_train_loader_iter = enumerate(source_train_loader)
                _, batch = source_train_loader_iter.__next__()
            images_source, label_panop_dict, shape, img_name_source = batch
            for key in label_panop_dict.keys():
                try:
                    label_panop_dict[key] = label_panop_dict[key].to(DEVICE)
                except:
                    pass

            # TODO: comment it later
            torch.cuda.synchronize(DEVICE)
            # source_dataloader_time = time.time() - start_time
            source_dataloader_time = perf_counter() - start_time
            print('source_dataloader_time = {:.3f}'.format(source_dataloader_time))

            start_time = perf_counter()
            perf_time['src_data'].append(source_dataloader_time)

            images_source = images_source.to(DEVICE)
            # data_time.update(time.time() - start_time) # TODO: uncomment
            pred = model(images_source)
            pred = get_module(model, cfg.DISTRIBUTED).upsample_predictions(pred, input_shape_source)

            # TODO: comment it later
            torch.cuda.synchronize(DEVICE)
            forward_pass_time = perf_counter() - start_time
            # forward_pass_time = time.time() - start_time
            print('forward_pass_time = {:.3f}'.format(forward_pass_time))
            perf_time['forward_pass_data'].append(forward_pass_time)
            start_time = perf_counter()

            loss_semantic, loss_center, loss_offset, loss_depth = \
                get_module(model, cfg.DISTRIBUTED).compute_loss(
                    pred,
                    label_panop_dict,
                    criterion_semseg,
                    criterion_center,
                    criterion_offset,
                    criterion_depth,
                    DEVICE,
                )
            loss = None
            batch_size = images_source.size(0)
            if not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
                loss = loss_semantic
            elif not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
                loss = loss_semantic + loss_depth
            elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
                loss = loss_semantic + loss_center + loss_offset
            elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
                loss = loss_semantic + loss_center + loss_offset + loss_depth
            # loss.backward()
            # loss_meter_dict['loss'].update(loss.detach().cpu().item(), batch_size)

            semantic_pred_source = None
            center_pred_source = None
            offset_pred_source = None
            depth_pred_source = None
            if not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
                loss_meter_dict['loss_semantic'].update(loss_semantic.detach().cpu().item(), batch_size)
                semantic_pred_source = pred['semantic']
            elif not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
                loss_meter_dict['loss_semantic'].update(loss_semantic.detach().cpu().item(), batch_size)
                loss_meter_dict['loss_depth'].update(loss_depth.detach().cpu().item(), batch_size)
                semantic_pred_source = pred['semantic']
                depth_pred_source = pred['depth']
            elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
                loss_meter_dict['loss_semantic'].update(loss_semantic.detach().cpu().item(), batch_size)
                loss_meter_dict['loss_center'].update(loss_center.detach().cpu().item(), batch_size)
                loss_meter_dict['loss_offset'].update(loss_offset.detach().cpu().item(), batch_size)
                semantic_pred_source = pred['semantic']
                center_pred_source = pred['center']
                offset_pred_source = pred['offset']
            elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
                loss_meter_dict['loss_semantic'].update(loss_semantic.detach().cpu().item(), batch_size)
                loss_meter_dict['loss_center'].update(loss_center.detach().cpu().item(), batch_size)
                loss_meter_dict['loss_offset'].update(loss_offset.detach().cpu().item(), batch_size)
                loss_meter_dict['loss_depth'].update(loss_depth.detach().cpu().item(), batch_size)
                semantic_pred_source = pred['semantic']
                center_pred_source = pred['center']
                offset_pred_source = pred['offset']
                depth_pred_source = pred['depth']

            # TODO: comment it later
            torch.cuda.synchronize(DEVICE)
            supervised_loss_time = perf_counter() - start_time
            # supervised_loss_time = time.time() - start_time
            print('supervised_loss_time = {:.3f}'.format(supervised_loss_time))
            start_time = perf_counter()
            perf_time['sup_loss_data'].append(supervised_loss_time)

            instance_source = None
            if cfg.APPROACH_TYPE == 'DANDA':
                semantic_pred_src = torch.argmax(semantic_pred_source.detach(), dim=1, keepdim=True)
                semantic_pred_src = semantic_pred_src.squeeze(0)  # TODO - by doing this I restricting this model trainig with batch-size=1
                # TODO: for training with batch-size > 1, I need to update the code in ctrl/model_panop/inst_seg.py
                instance_source, center_source = inst_seg_source(semantic_pred_src, center_pred_source.detach(), offset_pred_source.detach())

            # DACS training - if true
            if cfg.TRAIN.TRAIN_WITH_DACS:
                try:
                    _, batch = target_train_loader_iter.__next__()
                except StopIteration:
                    target_train_loader_iter = enumerate(target_train_loader)
                    _, batch = target_train_loader_iter.__next__()
                images_target, label_panop_dict_target, shape_target, img_name_target = batch
                images_target = images_target.to(DEVICE)

                # TODO: comment it later
                torch.cuda.synchronize(DEVICE)
                target_dataloader_time = perf_counter() - start_time
                # target_dataloader_time = time.time() - start_time
                print('target_dataloader_time = {:.3f}'.format(target_dataloader_time))
                start_time = perf_counter()
                perf_time['tar_data'].append(target_dataloader_time)

                kwargs_dacs = dict(
                    cfg=cfg,
                    images_source=images_source,
                    images_target=images_target,
                    input_shape_target=input_shape_target,
                    DEVICE=DEVICE,
                    model=model,
                    ema_model=ema_model,
                    label_panop_dict=label_panop_dict,
                    dacs_unlabeled_loss_semantic=dacs_unlabeled_loss_semantic,
                    i_iter=i_iter,
                    current_epoch=current_epoch,
                    img_name_source=img_name_source,
                    img_name_target=img_name_target,
                )
                dacs_unlabeled_semantic_loss = train_dacs_one_iter(**kwargs_dacs)

                # TODO: comment it later
                torch.cuda.synchronize(DEVICE)
                unlabeled_train_time =  perf_counter() - start_time
                # unlabeled_train_time = time.time() - start_time
                print('unlabeled_train_time = {:.3f}'.format(unlabeled_train_time))
                start_time = perf_counter()
                perf_time['unlabeled_train_time'].append(unlabeled_train_time)

                loss = loss + dacs_unlabeled_semantic_loss
                loss.backward()
                loss_meter_dict['loss'].update(loss.detach().cpu().item(), batch_size)
                loss_meter_dict['loss_dacs_unlabeled_semantic'].update(dacs_unlabeled_semantic_loss.detach().cpu().item(), batch_size)



            # adversarial training for generator
            # images_target = None
            # semantic_pred_target = None
            # center_pred_target = None
            # offset_pred_target = None
            # label_panop_dict_target = None
            # depth_pred_target = None
            # if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
            #     try:
            #         _, batch = target_train_loader_iter.__next__()
            #     except StopIteration:
            #         target_train_loader_iter = enumerate(target_train_loader)
            #         _, batch = target_train_loader_iter.__next__()
            #     images_target, label_panop_dict_target, shape_target, img_name_target = batch
            #     images_target = images_target.to(DEVICE)
            #     pred_target = model(images_target, domain_label=target_label)
            #     # pred_target = model(images_target)
            #     pred_target = get_module(model, cfg.DISTRIBUTED).upsample_predictions(pred_target, input_shape_target)
            #     # semantic_pred_target = pred_target['semantic']
            #     # center_pred_target = pred_target['center']
            #     # offset_pred_target = pred_target['offset']
            #     if not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_target = pred_target['semantic']
            #     elif not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_target = pred_target['semantic']
            #         depth_pred_target = pred_target['depth']
            #     elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_target = pred_target['semantic']
            #         center_pred_target = pred_target['center']
            #         offset_pred_target = pred_target['offset']
            #     elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_target = pred_target['semantic']
            #         center_pred_target = pred_target['center']
            #         offset_pred_target = pred_target['offset']
            #         depth_pred_target = pred_target['depth']
            #     instance_target = None
            #     if cfg.APPROACH_TYPE == 'DANDA':
            #         semantic_pred_tar = torch.argmax(semantic_pred_target.detach(), dim=1, keepdim=True)
            #         semantic_pred_tar = semantic_pred_tar.squeeze(0)
            #         instance_target, center_target = inst_seg_target(semantic_pred_tar, center_pred_target.detach(), offset_pred_target.detach())
            #         d_out, d_out2nd = disc_fwd_pass_danda(cfg, discriminator, discriminator2nd, instance_target, semantic_pred_target,
            #                                     center_pred_target, depth_pred_target.detach(), DEVICE)
            #     else:
            #         d_out, d_out2nd = disc_fwd_pass(cfg, discriminator, discriminator2nd, semantic_pred_target, center_pred=center_pred_target,
            #                                         offset_pred=offset_pred_target, depth_pred=depth_pred_target, mode=cfg.ADV_FEATURE_MODE)
            #     loss_adv_gen = criterion_disc(d_out, source_label)
            #     loss_meter_dict['loss_adv_gen'].update(loss_adv_gen.detach().cpu().item(), batch_size)
            #     if cfg.ENABLE_DISCRIMINATOR_2ND:
            #         loss_adv_gen2nd = criterion_disc2nd(d_out2nd, source_label)
            #         loss_meter_dict['loss_adv_gen2nd'].update(loss_adv_gen2nd.detach().cpu().item(), batch_size)
            #     loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_gen + cfg.TRAIN.LAMBDA_ADV_MAIN_2ND * loss_adv_gen2nd
            #     loss.backward()
            #     # Train discriminator networks
            #     # enable training mode on discriminator networks
            #     for param in discriminator.parameters():
            #         param.requires_grad = True
            #     if cfg.ENABLE_DISCRIMINATOR_2ND:
            #         for param in discriminator2nd.parameters():
            #             param.requires_grad = True
            #     # train with source
            #     if not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_source = semantic_pred_source.detach()
            #     elif not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_source = semantic_pred_source.detach()
            #         depth_pred_source = depth_pred_source.detach()
            #     elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_source = semantic_pred_source.detach()
            #         center_pred_source = center_pred_source.detach()
            #         offset_pred_source = offset_pred_source.detach()
            #     elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_source = semantic_pred_source.detach()
            #         center_pred_source = center_pred_source.detach()
            #         offset_pred_source = offset_pred_source.detach()
            #         depth_pred_source = depth_pred_source.detach()
            #     # d_out = discriminator(prob_2_entropy(F.softmax(semantic_pred_source, dim=1)) * depth_pred_source)
            #     if cfg.APPROACH_TYPE == 'DANDA':
            #         d_out, d_out2nd = disc_fwd_pass_danda(cfg, discriminator, discriminator2nd, instance_source, semantic_pred_source,
            #                                               center_pred_source, depth_pred_source, DEVICE)
            #     else:
            #         d_out, d_out2nd = disc_fwd_pass(cfg, discriminator, discriminator2nd, semantic_pred_source, center_pred=center_pred_source,
            #                           offset_pred=offset_pred_source, depth_pred=depth_pred_source, mode=cfg.ADV_FEATURE_MODE)
            #     loss_d1st = criterion_disc(d_out, source_label)
            #     # loss_d = loss_d
            #     if cfg.ENABLE_DISCRIMINATOR_2ND:
            #         loss_d2nd = criterion_disc2nd(d_out2nd, source_label)
            #     loss_d = loss_d1st + loss_d2nd
            #     loss_d.backward()
            #     # train with target
            #     if not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_target = semantic_pred_target.detach()
            #     elif not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_target = semantic_pred_target.detach()
            #         depth_pred_target = depth_pred_target.detach()
            #     elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_target = semantic_pred_target.detach()
            #         center_pred_target = center_pred_target.detach()
            #         offset_pred_target = offset_pred_target.detach()
            #     elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #         semantic_pred_target = semantic_pred_target.detach()
            #         center_pred_target = center_pred_target.detach()
            #         offset_pred_target = offset_pred_target.detach()
            #         depth_pred_target = depth_pred_target.detach()
            #     # d_out = discriminator(prob_2_entropy(F.softmax(semantic_pred_target, dim=1)) * depth_pred_target)
            #     if cfg.APPROACH_TYPE == 'DANDA':
            #         d_out, d_out2nd = disc_fwd_pass_danda(cfg, discriminator, discriminator2nd, instance_target,
            #                                               semantic_pred_target, center_pred_target, depth_pred_target, DEVICE)
            #     else:
            #         d_out, d_out2nd = disc_fwd_pass(cfg, discriminator, discriminator2nd, semantic_pred_target, center_pred=center_pred_target,
            #                           offset_pred=offset_pred_target, depth_pred=depth_pred_target, mode=cfg.ADV_FEATURE_MODE)
            #     loss_d1st = criterion_disc(d_out, target_label)
            #     loss_meter_dict['loss_disc'].update(loss_d1st.detach().cpu().item(), batch_size)
            #     # loss_d = loss_d
            #     if cfg.ENABLE_DISCRIMINATOR_2ND:
            #         loss_d2nd = criterion_disc2nd(d_out2nd, target_label)
            #         loss_meter_dict['loss_disc2nd'].update(loss_d2nd.detach().cpu().item(), batch_size)
            #     loss_d = loss_d1st + loss_d2nd
            #     loss_d.backward()

            optimizer.step()

            # TODO: comment it later
            torch.cuda.synchronize(DEVICE)
            loss_bacward_time = perf_counter() - start_time
            # loss_bacward_time = time.time() - start_time
            print('loss_bacward_time = {:.3f}'.format(loss_bacward_time))
            start_time = perf_counter()
            perf_time['backward_pass_time'].append(loss_bacward_time)

            if cfg.TRAIN.TRAIN_WITH_DACS:
                alpha_teacher = 0.99
                ema_model = update_ema_variables(cfg, ema_model=ema_model, model=model, alpha_teacher=alpha_teacher, iteration=i_iter)

            # TODO: comment it later
            torch.cuda.synchronize(device=0)
            mean_teacher_update_time = perf_counter() - start_time
            # loss_bacward_time = time.time() - start_time
            print('mean_teacher_update_time = {:.3f}'.format(mean_teacher_update_time))
            perf_time['mean_teacher_update_time'].append(mean_teacher_update_time)

            if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
                optimizer_disc.step()
                if cfg.ENABLE_DISCRIMINATOR_2ND:
                    optimizer_disc2nd.step()

            # update learning rate
            lr = 0.0
            lrd = 0.0
            lrd2nd = 0.0
            if cfg.PANOPTIC_DEEPLAB_STYLE_LR:
                lr = optimizer.param_groups[best_param_group_id]["lr"]
                lr_scheduler.step()
                if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
                    lr_scheduler_disc.step()
            elif cfg.DADA_STYLE_LR:
                lr = adjust_learning_rate(optimizer, i_iter, cfg, cfg.DEBUG)
                if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR:
                    lrd = adjust_learning_rate_disc(optimizer_disc, i_iter, cfg, cfg.DEBUG)
                if not cfg.TRAIN_ONLY_SOURCE and cfg.ENABLE_DISCRIMINATOR_2ND:
                    lrd2nd = adjust_learning_rate_disc(optimizer_disc2nd, i_iter, cfg, cfg.DEBUG)

            batch_time.update(time.time() - start_time)

            # print current losses
            if i_iter == 0 or (i_iter+1) % cfg.TRAIN.DISPLAY_LOSS_RATE == 0:
                msg = '[{0}/{1}] GEN_BASE_LR: {2:.7f} DISC_BASE_LR: {3:.7f}\t DISC_BASE_LR_2ND: {4:.7f}\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.\
                    format(i_iter + 1, max_iter, lr, lrd, lrd2nd, batch_time=batch_time, data_time=data_time)


                msg += get_loss_info_str(loss_meter_dict)
                logger.info(msg)

            # tensorboard loss updates
            if viz_tensorboard:
                if comm.is_main_process():
                    log_losses_tensorboard_cvpr2022(writer, loss_meter_dict, i_iter)
                # log_losses_tensorboard_cvpr2021(writer, loss_meter_dict, i_iter)

            # tensor board visualization
            # if i_iter == 0 or (i_iter + 1) % cfg.TRAIN.TENSORBOARD_VIZRATE == 0:
            #     if comm.is_main_process():
            #         vis_sem_pred = None
            #         vis_dep_pred = None
            #         vis_cen_pred = None
            #         vis_ofs_pred = None
            #         if not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #             vis_sem_pred = True
            #         elif not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #             vis_sem_pred = True
            #             vis_dep_pred = True
            #         elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #             vis_sem_pred = True
            #             vis_cen_pred = True
            #             vis_ofs_pred = True
            #         elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
            #             vis_sem_pred = True
            #             vis_cen_pred = True
            #             vis_ofs_pred = True
            #             vis_dep_pred = True
            #         logger.info('predictions and gt labels for cityscapes eval image id {} are displayed in tensorbaord ... '.format(i_iter + 1))
            #         draw_in_tensorboard(writer, images_source, i_iter, semantic_pred_source,
            #                             center_pred_source, offset_pred_source, num_classes, label_panop_dict,
            #                             source_train_loader.dataset, vis_sem_pred, vis_dep_pred, vis_cen_pred, vis_ofs_pred, 'source', 0)
            #         if not cfg.TRAIN_ONLY_SOURCE:
            #             draw_in_tensorboard(writer, images_target, i_iter, semantic_pred_target,
            #                                 center_pred_target, offset_pred_target, num_classes, label_panop_dict_target,
            #                                 target_train_loader.dataset, vis_sem_pred, vis_dep_pred, vis_cen_pred, vis_ofs_pred, 'target', 0)

            # save checkpoint
            if i_iter == 0 or (i_iter + 1) % cfg.TRAIN.SAVE_PRED_EVERY == 0:
                if comm.is_main_process():
                    save_dict = {
                        'iter': i_iter,
                        'max_iter': cfg.TRAIN.MAX_ITER,
                        'model_state_dict': get_module(model, cfg.DISTRIBUTED).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ema_model_state_dict': get_module(ema_model, cfg.DISTRIBUTED).state_dict() if cfg.TRAIN.TRAIN_WITH_DACS else None,
                        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                        'disc_state_dict': discriminator.state_dict() if discriminator else None,
                        'disc_optim_state_dict': optimizer_disc.state_dict() if optimizer_disc else None,
                        'loss': loss,
                    }
                    checkpoint_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, 'model_{}_{}.pth'.format(i_iter + 1, current_epoch))
                    torch.save(save_dict, checkpoint_path)
                    logger.info("Saving the checkpoint at: {}".format(checkpoint_path))

            # display the model checkpoint path
            if (i_iter + 1) % 200 == 0:
                logger.info('')
                logger.info('cfg.BSUB_SCRIPT_FNAME: {}'.format(cfg.BSUB_SCRIPT_FNAME))
                logger.info('cfg.TRAIN.SNAPSHOT_DIR: {}'.format(cfg.TRAIN.SNAPSHOT_DIR))
                logger.info('')

            # evaluate and save best chekpoint under best_model folder
            if cfg.ACTIVATE_SEMANITC_EVAL:
                if ( not cfg.DISTRIBUTED and ((i_iter + 1) % cfg.TRAIN.EVAL_EVERY == 0) ) \
                        or (not cfg.DISTRIBUTED and ((i_iter + 1) == cfg.TRAIN.MAX_ITER) ):
                    set_nets_mode(cfg, model, discriminator, discriminator2nd, 'eval')
                    cIoU, mIoU = eval_model(model, target_val_loader, DEVICE, cfg)
                    set_nets_mode(cfg, model, discriminator, discriminator2nd, 'train')
                    if best_miou < mIoU:
                        best_miou = mIoU
                        if comm.is_main_process():
                            save_dict = {
                                'iter': i_iter,
                                'max_iter': cfg.TRAIN.MAX_ITER,
                                'model_state_dict': get_module(model, cfg.DISTRIBUTED).state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                                'disc_state_dict': discriminator.state_dict() if discriminator else None,
                                'disc_optim_state_dict': optimizer_disc.state_dict() if optimizer_disc else None,
                                'loss': loss,
                            }
                            checkpoint_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL, 'model_{}_{}.pth'.format(i_iter+1, current_epoch))
                            torch.save(save_dict, checkpoint_path)
                            best_checkpoint_path = checkpoint_path
                            logger.info("Saving the best checkpoint at: {}".format(checkpoint_path))
                            name_classes = target_val_loader.dataset.class_names
                            for ind_class in range(cfg.NUM_CLASSES):
                                logger.info(name_classes[ind_class] + '\t' + str(round(cIoU[ind_class] * 100, 2)))
                            logger.info('*** BEST mIoU: {} ***'.format(best_miou))
                    else:
                        logger.info('*** BEST mIoU: {} ***'.format(best_miou))
                        logger.info('*** BEST CHECKPOINT PATH : {}'.format(best_checkpoint_path))

            if cfg.ACTIVATE_PANOPTIC_EVAL:
                if (not cfg.DISTRIBUTED and ((i_iter + 1) % cfg.TRAIN.EVAL_EVERY == 0)) \
                        or (not cfg.DISTRIBUTED and ((i_iter + 1) == cfg.TRAIN.MAX_ITER)):
                    logger.info('')
                    logger.info('START: panoptic evaluation ...')
                    num_imgs_to_dispaly = 5
                    img_ids = [random.randrange(0, target_test_nsamp, 1) for i in range(num_imgs_to_dispaly)]
                    logger.info('Predictions for these images will be displayed in tensoboard:')
                    logger.info(img_ids)
                    set_nets_mode(cfg, model, discriminator, discriminator2nd, 'eval')
                    eval_panoptic(cfg, model, panop_eval_folder_dict, panop_eval_writer,
                                  i_iter, logger, DEVICE, target_val_loader, img_ids)
                    set_nets_mode(cfg, model, discriminator, discriminator2nd, 'train')
                    # TODO: remove here the eval folder
                    strCmd2 = 'rm -r ' + ' ' + panop_eval_root_folder
                    os.system(strCmd2)
                    logger.info('Removing the panoptic evaluation root folder ...')
                    logger.info('panop_eval_root_folder: {}'.format(panop_eval_root_folder))
                    logger.info('END: panoptic evaluation ...')
                    logger.info('Creating the folder and sub folders for next panoptic evaluation ...')
                    # panop_eval_root_folder has a unique name and generated using current time stamp
                    panop_eval_folder_dict, panop_eval_root_folder = create_panop_eval_dirs(cfg.TRAIN.SNAPSHOT_DIR, logger)
                    panop_eval_writer = SummaryWriter(log_dir=panop_eval_folder_dict['tensorboard'])
                    logger.info('')

        # TODO: comment it later
        src_data = np.mean(perf_time['src_data'])
        forward_pass_data = np.mean(perf_time['forward_pass_data'])
        sup_loss_data = np.mean(perf_time['sup_loss_data'])
        tar_data = np.mean(perf_time['tar_data'])
        unlabeled_train_time = np.mean(perf_time['unlabeled_train_time'])
        backward_pass_time = np.mean(perf_time['backward_pass_time'])
        mean_teacher_update_time = np.mean(perf_time['mean_teacher_update_time'])
        print('************** MEAN **********************')
        print('src_data: {:.3f}'.format(src_data))
        print('forward_pass_data: {:.3f}'.format(forward_pass_data))
        print('sup_loss_data: {:.3f}'.format(sup_loss_data))
        print('tar_data: {:.3f}'.format(tar_data))
        print('unlabeled_train_time: {:.3f}'.format(unlabeled_train_time))
        print('backward_pass_time: {:.3f}'.format(backward_pass_time))
        print('mean_teacher_update_time: {:.3f}'.format(mean_teacher_update_time))

    except Exception:
        logger.exception("Exception during training:")
        raise
    finally:
        if comm.is_main_process():
            save_dict = {
                'iter': cfg.TRAIN.MAX_ITER,
                'max_iter': cfg.TRAIN.MAX_ITER,
                'model_state_dict': get_module(model, cfg.DISTRIBUTED).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                'disc_state_dict': discriminator.state_dict() if discriminator else None,
                'disc_optim_state_dict': optimizer_disc.state_dict() if optimizer_disc else None,
            }
            checkpoint_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, 'model_{}_{}.pth'.format(cfg.TRAIN.MAX_ITER, current_epoch))
            torch.save(save_dict, checkpoint_path)
            logger.info("Saving the checkpoint at: {}".format(checkpoint_path))
        logger.info("Training finished.")





















