from ctrl.utils.common_config_panop import get_criterion_panop, get_optimizer_dacs
from ctrl.utils.train_utils import log_output_paths
import argparse
from ctrl.utils.panoptic_deeplab.logger import setup_logger
import logging
from ctrl.utils.panoptic_deeplab import comm
from ctrl.utils.panoptic_deeplab.utils import to_cuda
from ctrl.config_panop import config, update_config
from ctrl.machine_specific_paths import msp
import os
from torch.utils import data
from ctrl.utils.panoptic_deeplab.env import seed_all_rng
from ctrl.dacs_old.model.dacs_model import DACSModel
from ctrl.dacs_old.utils.train_uda_scripts import create_ema_model
from torch.hub import load_state_dict_from_url as load_url
from ctrl.utils.panoptic_deeplab.utils import get_loss_info_str
from ctrl.dacs_old.utils import transformmasks
from ctrl.dacs_old.utils.train_uda_scripts import strongTransform
from ctrl.utils.panoptic_deeplab.utils import get_module
import torch
import random
from ctrl.dacs_old.utils.train_uda_scripts import save_image
import ctrl.dacs_old.utils.palette as palette
from torch.utils.tensorboard import SummaryWriter
from ctrl.eval_semantics import eval_model
from ctrl.utils.panoptic_deeplab import AverageMeter
from collections import OrderedDict
from ctrl.dacs_old.utils.train_uda_scripts import update_ema_variables
from ctrl.utils.train_utils import log_losses_tensorboard_cvpr2022
from ctrl.utils.train_utils import adjust_learning_rate
import numpy as np
from ctrl.dataset.synthia_dacs import SYNTHIADataSetDepth
from ctrl.dataset.cityscapes_dacs import CityscapesDataSet


class Learning_Rate_Object(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


def _worker_init_fn_corda_style(worker_id):
    # Taken from https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    print('*** _worker_init_fn_corda_style() : worker_id: {} worker_seed: {} ***'.format(worker_id, worker_seed))


def setup_seeds_corda_style(seed, logger):
    logger.info('*** def setup_seeds_corda_style(seed={}) ***'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn related setting - good for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def _worker_init_fn_panoptic_deeplab(worker_id):
    randId = np.random.randint(2 ** 31)
    seed = randId + worker_id
    print('**** [randId: {} + worker_id: {}] = seed: {} ***'.format(randId, worker_id, seed))
    seed_all_rng(seed)


def get_exp_fname():
    exp_file = 'ctrl/yml-config-panop/bsub_euler_expid7_5_0_1.yml'  # original DACS
    return exp_file


def parse_args():
    cfg_file = get_exp_fname()  # this expression is bypassed when setting the yml file via --cfg option
    parser = argparse.ArgumentParser(description='Train CVPR2022 models')
    parser.add_argument('--debug', type=str, default='True', help='debug flag')
    parser.add_argument('--yml_fname', type=str, default=cfg_file, help='experiment config_panop yml file name')
    parser.add_argument("--local_rank", type=int, default=0, help='main process id')
    parser.add_argument('opts', help="Modify config_panop options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config.DEBUG = True if args.debug == 'True' else False
    if config.DEBUG:
        data_root = msp[0]['data_root']
        ms_exp_root = msp[0]['machine_spec_exp_root']
        pimnet_path = msp[0]['imgnet_pretrained_model_path']
        exp_root = 'exp_root_debug_Sep22'
        phase_name = 'phase_debug_Sep22'
        sub_phase_name = 'sub_phase_debug_Sep22'
        args.opts = ["DATA_ROOT", data_root, "MS_EXP_ROOT", ms_exp_root, "EXP_ROOT", exp_root, "PHASE_NAME", phase_name, "SUB_PHASE_NAME", sub_phase_name]
    update_config(config, args)
    return args


# *** MAIN FUNCTION ***
def main():
    args = parse_args()
    cfg = config
    DEVICE = torch.device('cuda:{}'.format(args.local_rank))
    # setup panopitc-deeplab logger
    logger = logging.getLogger('ctrl')
    if not logger.isEnabledFor(logging.INFO):
        setup_logger(output=cfg.TRAIN_LOG_FNAME, distributed_rank=args.local_rank)
    logger.info('*********************************************')
    logger.info('cfg.BSUB_SCRIPT_FNAME:  {}'.format(cfg.BSUB_SCRIPT_FNAME))
    logger.info('*********************************************')
    logger.info('')
    log_output_paths(cfg)
    setup_seeds_corda_style(cfg.TRAIN.RANDOM_SEED, logger)
    model = DACSModel(cfg)
    model_params_saved = load_url(cfg.TRAIN.DACS_IMAGENET_COCO_PRETRAINED_MODEL)
    logger.info('ResNet Backbone is initialized with dacs_old imagenet coco pretrained weights')
    logger.info('from: {}'.format(cfg.TRAIN.DACS_IMAGENET_COCO_PRETRAINED_MODEL))
    model_params_current = model.state_dict().copy()
    for name, param in model_params_current.items():
        name_parts = name.split(".")
        name2 = ".".join(name_parts[1:])
        if name2 in model_params_saved and param.size() == model_params_saved[name2].size():
            model_params_current[name].copy_(model_params_saved[name2])
    model.load_state_dict(model_params_current)
    logger.info('cfg.MODEL_SUB_TYPE: {}'.format(cfg.MODEL_SUB_TYPE))
    logger.info('cfg.TRAIN.FREEZE_BN: {}'.format(cfg.TRAIN.FREEZE_BN))
    logger.info('***')
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if cfg.TRAIN.FREEZE_BN:
                module.requires_grad_(False)
            else:
                module.requires_grad_(True)
    ema_model = create_ema_model(model, cfg)
    model = model.to(DEVICE)
    ema_model = ema_model.to(DEVICE)
    # data loaders
    import numpy as np
    IMG_MEAN = np.array(cfg.TRAIN.IMG_MEAN, dtype=np.float32)
    source_train_dataset = SYNTHIADataSetDepth(
        root=cfg.DATA_DIRECTORY_SOURCE,
        list_path=cfg.DATA_LIST_SOURCE,
        set=cfg.TRAIN.SET_SOURCE,
        num_classes=cfg.NUM_CLASSES,
        max_iters=None,
        crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
        mean=IMG_MEAN,
        use_depth=cfg.USE_DEPTH,
        depth_processing=cfg.DEPTH_PROCESSING,
        cfg=cfg,
        joint_transform=None,
    )
    source_train_loader = data.DataLoader(
        source_train_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=_worker_init_fn_corda_style,
        drop_last=True,
    )
    target_train_dataset = CityscapesDataSet(
        root=cfg.DATA_DIRECTORY_TARGET,
        list_path=cfg.DATA_LIST_TARGET,
        set=cfg.TRAIN.SET_TARGET,
        info_path=cfg.TRAIN.INFO_TARGET,
        max_iters=None,
        crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
        mean=IMG_MEAN,
        joint_transform=None,
        cfg=cfg,
    )
    target_train_loader = data.DataLoader(
        target_train_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=_worker_init_fn_corda_style,  # OR _worker_init_fn_panoptic_deeplab
        drop_last=True,
    )
    target_test_dataset = CityscapesDataSet(
        root=cfg.DATA_DIRECTORY_TARGET,
        list_path=cfg.DATA_LIST_TARGET,
        set=cfg.TEST.SET_TARGET,
        info_path=cfg.TEST.INFO_TARGET,
        crop_size=cfg.TEST.INPUT_SIZE_TARGET,
        mean=cfg.TEST.IMG_MEAN,
        labels_size=cfg.TEST.OUTPUT_SIZE_TARGET,
        joint_transform=None,
        cfg=cfg,
    )
    target_val_loader = data.DataLoader(
        target_test_dataset,
        batch_size=1,  # cfg.TRAIN.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
    )
    source_train_nsamp = len(source_train_dataset)
    logger.info('{} : source train examples: {}'.format(cfg.SOURCE, source_train_nsamp))
    target_train_nsamp = len(target_train_dataset)
    target_test_nsamp = len(target_test_dataset)
    logger.info('{} : target train examples: {}'.format(cfg.TARGET, target_train_nsamp))
    logger.info('{} : target test examples: {}'.format(cfg.TARGET, target_test_nsamp))
    # get optimizer
    optimizer = get_optimizer_dacs(cfg, model)
    logger.info(optimizer)
    # get the lr scheduler
    lr_scheduler = None
    # get criterion
    criterion_dict = get_criterion_panop(cfg)
    to_cuda(criterion_dict, DEVICE)
    logger.info(criterion_dict)
    criterion_semseg = criterion_dict['semseg']
    dacs_unlabeled_loss_semantic = criterion_dict['dacs_unlabeled_loss_semantic']
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    model.train()
    ema_model.train()
    source_train_loader_iter = enumerate(source_train_loader)
    target_train_loader_iter = enumerate(target_train_loader)
    if source_train_nsamp > target_train_nsamp:
        iterInOneEpoch = int(source_train_nsamp / cfg.TRAIN.IMS_PER_BATCH)
    else:
        iterInOneEpoch = int(target_train_nsamp / cfg.TRAIN.IMS_PER_BATCH)
    num_train_epochs = int(cfg.TRAIN.MAX_ITER / iterInOneEpoch)
    logger.info('num iterations in one epoch: {}'.format(iterInOneEpoch))
    logger.info('total epoch to train: {}'.format(num_train_epochs))
    loss_meter_dict = OrderedDict()
    loss_meter_dict['loss_semantic'] = AverageMeter()
    loss_meter_dict['loss_dacs_unlabeled_semantic'] = AverageMeter()
    loss_meter_dict['loss'] = AverageMeter()
    max_iter = cfg.TRAIN.MAX_ITER
    best_miou = -1
    best_iter = 0
    best_checkpoint_path = ''
    logger.info('')
    logger.info('*** TRAINING STARTS HERE ***')
    logger.info('')
    start_iter = 0
    local_iter = 0
    current_epoch = 0
    input_size_source = (cfg.DATASET.RANDOM_CROP_DIM, cfg.DATASET.RANDOM_CROP_DIM)
    input_size_target = (cfg.DATASET.RANDOM_CROP_DIM, cfg.DATASET.RANDOM_CROP_DIM)
    input_shape_source = (input_size_source[1], input_size_source[0])
    input_shape_target = (input_size_target[1], input_size_target[0])
    try:
        for i_iter in range(start_iter, max_iter):
            # epoch count
            if (i_iter + 1) % iterInOneEpoch == 0:
                current_epoch += 1
            local_iter += 1
            optimizer.zero_grad()
            # source data loading
            try:
                _, batch = source_train_loader_iter.__next__()
            except StopIteration:
                source_train_loader_iter = enumerate(source_train_loader)
                _, batch = source_train_loader_iter.__next__()
            images_source, label_source, shape_source, img_name_source = batch
            images_source = images_source.to(DEVICE)
            label_source = label_source.to(DEVICE)
            pred = model(images_source)
            pred = get_module(model, cfg.DISTRIBUTED).upsample_predictions(pred, input_shape_source)
            loss_semantic = criterion_semseg(pred['semantic'], label_source) * 1.0
            batch_size = images_source.size(0)
            loss = loss_semantic
            loss_meter_dict['loss_semantic'].update(loss_semantic.detach().cpu().item(), batch_size)
            # target data loading
            try:
                _, batch = target_train_loader_iter.__next__()
            except StopIteration:
                target_train_loader_iter = enumerate(target_train_loader)
                _, batch = target_train_loader_iter.__next__()
            images_target, _, _, img_name_target = batch
            images_target = images_target.to(DEVICE)
            # *** DACS training ***
            # pseudo labels generation - forward pass through the teacher model (ema_model)
            pred_target = ema_model(images_target)
            pred_target = get_module(ema_model, cfg.DISTRIBUTED).upsample_predictions(pred_target, input_shape_target)
            logits_u_w = pred_target['semantic']
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)
            # once pseduo lable generated, now generate the masks for augmentation
            # which pixels from source are to be augmented to the target - for this we need to create mask
            MixMasks = []
            inputs_u_s = []
            for image_i in range(cfg.TRAIN.IMS_PER_BATCH):
                classes = torch.unique(label_source[image_i, :])  # get the GT label for the source image
                nclasses = classes.shape[0]
                classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)).long()]).to(DEVICE)
                MixMasks.append(transformmasks.generate_class_mask(label_source[image_i, :], classes).unsqueeze(0).to(DEVICE))
            # once mask is created, based on the masks
            # create cross domain synthetic images from the source and the target images
            strong_parameters = {}
            if cfg.TRAIN.DACS_UNLABELED_FLIP:
                strong_parameters["flip"] = random.randint(0, 1)
            else:
                strong_parameters["flip"] = 0
            if cfg.TRAIN.DACS_COLOR_JITTER:
                strong_parameters["ColorJitter"] = random.uniform(0, 1)
            else:
                strong_parameters["ColorJitter"] = 0
            if cfg.TRAIN.DACS_BLUR:
                strong_parameters["GaussianBlur"] = random.uniform(0, 1)
            else:
                strong_parameters["GaussianBlur"] = 0
            imC = 0
            for MixMask in MixMasks:
                strong_parameters["Mix"] = MixMask
                inputs_u_s_temp, _ = strongTransform(cfg, strong_parameters, data=torch.cat((images_source[imC, :].unsqueeze(0), images_target[imC, :].unsqueeze(0))))
                inputs_u_s.append(inputs_u_s_temp.squeeze(0))
                imC += 1
            # stack the cross-domain synthetic images
            # these are the input images to the model
            inputs_u_s = torch.stack(inputs_u_s, dim=0)
            # forward pass of the synthetic augmented images through the student model (model)
            pred_synthetic = model(inputs_u_s)
            pred_synthetic = get_module(model, cfg.DISTRIBUTED).upsample_predictions(pred_synthetic, input_shape_target)
            logits_u_s = pred_synthetic['semantic']
            # once mask is created, based on the masks
            # create cross domain synthetic label_source from the source and the target label_source
            imC = 0
            targets_u = []
            for MixMask in MixMasks:
                strong_parameters["Mix"] = MixMask
                _, targets_u_temp = strongTransform(cfg, strong_parameters, target=torch.cat((label_source[imC, :].unsqueeze(0), targets_u_w[imC, :].unsqueeze(0))))
                targets_u.append(targets_u_temp.squeeze(0))
                imC += 1
            # stack the cross-domain synthetic labels
            targets_u = torch.stack(targets_u, dim=0).long()
            # thresholding the pseudo labels
            pixel_weight = cfg.LOSS.DACS.UNLABELED_LOSS.PIXEL_WEIGHT
            if pixel_weight == "threshold_uniform":
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).to(DEVICE)
            elif pixel_weight == "threshold":
                pixelWiseWeight = max_probs.ge(0.968).float().to(DEVICE)
            elif pixel_weight == False:
                pixelWiseWeight = torch.ones(max_probs.shape).to(DEVICE)
            onesWeights = torch.ones((pixelWiseWeight.shape)).to(DEVICE)
            # compute the pixel wise weight for the loss computation
            # for source pixels all the weights are 1 as we know the GT
            # for  target pixels all the weights are based on the
            # probability scores of the predictions by the teacher network
            imC = 0
            pixel_wise_weight = []
            for MixMask in MixMasks:
                strong_parameters["Mix"] = MixMask
                _, pixelWiseWeightTemp = strongTransform(cfg, strong_parameters, target=torch.cat((onesWeights[imC, :].unsqueeze(0), pixelWiseWeight[imC, :].unsqueeze(0))))
                pixel_wise_weight.append(pixelWiseWeightTemp.squeeze(0))
                imC += 1
            # stack the cross-domain pixel wise weights for loss computation
            pixelWiseWeight = torch.stack(pixel_wise_weight, dim=0).to(DEVICE)
            consistency_loss = cfg.LOSS.DACS.UNLABELED_LOSS.CONSISTENCY_LOSS
            consistency_weight = cfg.LOSS.DACS.UNLABELED_LOSS.CONSISTENCY_WEIGHT
            dacs_unlabeled_semantic_loss = None
            if consistency_loss == 'MSE':
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                dacs_unlabeled_semantic_loss = consistency_weight * unlabeled_weight * dacs_unlabeled_loss_semantic(logits_u_s, pseudo_label)
            elif consistency_loss == 'CE':
                dacs_unlabeled_semantic_loss = consistency_weight * dacs_unlabeled_loss_semantic(logits_u_s, targets_u, pixelWiseWeight)
            # Saves two mixed images and the corresponding prediction
            # save_image(cfg, image, input_fname, pred_fname, id, palette, img_name_source, img_name_target):
            if i_iter == 0 or (i_iter + 1) % cfg.TRAIN.DACS_SAVE_IMG_EVERY == 0:
                print('Saved two cross-domain mixed images and the corresponding predictions:')
                for bi in range(cfg.TRAIN.IMS_PER_BATCH):
                    input_fname = 'Input-BatchId_{}-Epoch_{}-Iter_{}.png'.format(bi, current_epoch, i_iter)
                    print(input_fname)
                    save_image(cfg, inputs_u_s[bi].cpu(), input_fname, '', palette.CityScpates_palette)

                _, pred_u_s = torch.max(logits_u_s, dim=1)
                for bi in range(cfg.TRAIN.IMS_PER_BATCH):
                    input_fname = 'Pred-BatchId_{}-Epoch_{}-Iter_{}.png'.format(bi, current_epoch, i_iter)
                    print(input_fname)
                    save_image(cfg, pred_u_s[bi].cpu(), '', input_fname, palette.CityScpates_palette)
            loss = loss + dacs_unlabeled_semantic_loss
            # backward pass and gradient update
            loss.backward()
            optimizer.step()
            # update lr
            lr = adjust_learning_rate(optimizer, i_iter, cfg, cfg.DEBUG)
            loss_meter_dict['loss'].update(loss.detach().cpu().item(), batch_size)
            loss_meter_dict['loss_dacs_unlabeled_semantic'].update(dacs_unlabeled_semantic_loss.detach().cpu().item(), batch_size)
            # ema modelupdate
            alpha_teacher = 0.99
            ema_model = update_ema_variables(cfg, ema_model=ema_model, model=model, alpha_teacher=alpha_teacher, iteration=i_iter)
            # print current losses
            if i_iter == 0 or (i_iter + 1) % cfg.TRAIN.DISPLAY_LOSS_RATE == 0:
                msg = '[{0}/{1}] GEN_BASE_LR: {2:.7f}\t'. \
                    format(i_iter + 1, max_iter, lr)
                msg += get_loss_info_str(loss_meter_dict)
                logger.info(msg)
            if viz_tensorboard:
                log_losses_tensorboard_cvpr2022(writer, loss_meter_dict, i_iter)
            # save checkpoint
            if i_iter == 0 or (i_iter + 1) % cfg.TRAIN.SAVE_PRED_EVERY == 0:
                save_dict = {
                    'iter': i_iter,
                    'max_iter': cfg.TRAIN.MAX_ITER,
                    'model_state_dict': get_module(model, cfg.DISTRIBUTED).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_model_state_dict': get_module(ema_model, cfg.DISTRIBUTED).state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
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
                logger.info('*** BEST mIoU: {} ***'.format(best_miou))
                logger.info('*** BEST iter: {} ***'.format(best_iter))
            # evaluate and save best chekpoint under best_model folder
            if True:
                if (not cfg.DISTRIBUTED and ((i_iter + 1) % cfg.TRAIN.EVAL_EVERY == 0)) \
                        or (not cfg.DISTRIBUTED and ((i_iter + 1) == cfg.TRAIN.MAX_ITER)) \
                        or (not cfg.DISTRIBUTED and ((i_iter + 1) == 1000)):
                    model.eval()
                    cIoU, mIoU = eval_model(model, target_val_loader, DEVICE, cfg, mode='dacs')
                    model.train()
                    if best_miou < mIoU:
                        best_miou = mIoU
                        best_iter = i_iter + 1
                        if comm.is_main_process():
                            save_dict = {
                                'iter': i_iter,
                                'max_iter': cfg.TRAIN.MAX_ITER,
                                'model_state_dict': get_module(model, cfg.DISTRIBUTED).state_dict(),
                                'ema_model_state_dict': get_module(ema_model, cfg.DISTRIBUTED).state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                                'loss': loss,
                            }
                            checkpoint_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR_BESTMODEL, 'model_{}_{}.pth'.format(i_iter + 1, current_epoch))
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
            }
            checkpoint_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, 'model_{}_{}.pth'.format(cfg.TRAIN.MAX_ITER, current_epoch))
            torch.save(save_dict, checkpoint_path)
            logger.info("Saving the checkpoint at: {}".format(checkpoint_path))
        logger.info("Training finished.")


if __name__ == "__main__":
    main()



