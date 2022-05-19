# START OF THE FUNCTION DEFINITION
import numpy as np
from ctrl.dacs_old.utils import transformmasks
from ctrl.dacs_old.utils.train_uda_scripts import strongTransform
from ctrl.utils.panoptic_deeplab.utils import get_module
import torch
import random
from ctrl.dacs_old.utils.train_uda_scripts import save_image
import ctrl.dacs_old.utils.palette as palette

def train_dacs_one_iter(cfg, images_source, images_target, input_shape_target, DEVICE,
                        model, ema_model, label_panop_dict, dacs_unlabeled_loss_semantic,
                        i_iter, current_epoch, img_name_source, img_name_target):

    # pseudo labels generation - forward pass through the teacher model (ema_model)
    pred_target = ema_model(images_target)
    pred_target = get_module(ema_model, cfg.DISTRIBUTED).upsample_predictions(pred_target, input_shape_target)
    logits_u_w = None
    if not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
        logits_u_w = pred_target['semantic']
    elif not cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
        logits_u_w = pred_target['semantic']
        depth_pred_target = pred_target['depth']
    elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and not cfg.TRAIN.TRAIN_DEPTH_BRANCH:
        logits_u_w = pred_target['semantic']
        center_pred_target = pred_target['center']
        offset_pred_target = pred_target['offset']
    elif cfg.TRAIN.TRAIN_INSTANCE_BRANCH and cfg.TRAIN.TRAIN_DEPTH_BRANCH:
        logits_u_w = pred_target['semantic']
        center_pred_target = pred_target['center']
        offset_pred_target = pred_target['offset']
        depth_pred_target = pred_target['depth']
    pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
    max_probs, targets_u_w = torch.max(pseudo_label, dim=1)

    # once pseduo lable generated, now generate the masks for augmentation
    # which pixels from source are to be augmented to the target - for this we need to create mask
    MixMasks = []
    inputs_u_s = []
    labels = label_panop_dict['semantic']
    for image_i in range(cfg.TRAIN.IMS_PER_BATCH):
        classes = torch.unique(labels[image_i, :])  # get the GT label for the source image
        nclasses = classes.shape[0]
        classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)).long()]).to(DEVICE)
        MixMasks.append(transformmasks.generate_class_mask(labels[image_i, :], classes).unsqueeze(0).to(DEVICE))

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

    # forward pass of the synthetic augmented images
    # through the student model (model)
    pred_synthetic = model(inputs_u_s)
    pred_synthetic = get_module(model, cfg.DISTRIBUTED).upsample_predictions(pred_synthetic, input_shape_target)
    logits_u_s = pred_synthetic['semantic']

    # once mask is created, based on the masks
    # create cross domain synthetic labels from the source and the target labels
    imC = 0
    targets_u = []
    for MixMask in MixMasks:
        strong_parameters["Mix"] = MixMask
        _, targets_u_temp = strongTransform(cfg, strong_parameters, target=torch.cat((labels[imC, :].unsqueeze(0), targets_u_w[imC, :].unsqueeze(0))))
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
    # for  target pixels all the weights are based on the probability scores of the predictions by the teacher network
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
    L_u = None
    if consistency_loss == 'MSE':
        unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
        L_u = consistency_weight * unlabeled_weight * dacs_unlabeled_loss_semantic(logits_u_s, pseudo_label)
    elif consistency_loss == 'CE':
        L_u = consistency_weight * dacs_unlabeled_loss_semantic(logits_u_s, targets_u, pixelWiseWeight)

    # Saves two mixed images and the corresponding prediction
    # save_image(cfg, image, input_fname, pred_fname, id, palette, img_name_source, img_name_target):
    if i_iter+1 == 1 or i_iter+1 == 10 or (i_iter+1) % cfg.TRAIN.DACS_SAVE_IMG_EVERY == 0:
        for i in range(cfg.TRAIN.IMS_PER_BATCH):
            str1 = img_name_source[i].split('.')
            str2 = img_name_target[i].split('.')
            input_fname = 'Input-Source-{}-Target-{}-Epoch_{}-Iter_{}.png'.format(str1[0], str2[0], current_epoch, i_iter)
            save_image(cfg, inputs_u_s[i].cpu(), input_fname, '', palette.CityScpates_palette)

        _, pred_u_s = torch.max(logits_u_s, dim=1)
        for i in range(cfg.TRAIN.IMS_PER_BATCH):
            str1 = img_name_source[i].split('.')
            str2 = img_name_target[i].split('.')
            pred_fname = 'Pred-Source-{}-Target-{}-Epoch_{}-Iter_{}.png'.format(str1[0], str2[0], current_epoch, i_iter)
            save_image(cfg, pred_u_s[i].cpu(), '', pred_fname, palette.CityScpates_palette)


    return L_u
