import torch
from ctrl.utils.train_utils import per_class_iu, fast_hist
import numpy as np
from torch import nn
import logging


def eval_model(model, test_loader, device, cfg, mode=None):
    logger = logging.getLogger(__name__)
    DEBUG = cfg.DEBUG
    # EXP_SETUP = cfg.EXP_SETUP
    # print('class list:')
    # print(test_loader.dataset.class_names)
    test_iter = iter(test_loader)
    # str_target_dataset_name = EXP_SETUP.split('_')[2]

    if 'Cityscapes' in cfg.TARGET:
        fixed_test_size = True
    elif 'Mapillary' in cfg.TARGET:
        fixed_test_size = False

    # if not cfg.TEST.OUTPUT_SIZE_TARGET:
    #     fixed_test_size = False
    # else:
    #     fixed_test_size = True

    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    num_val_samples = len(test_loader)
    logger.info('number of val samples : {}'.format(num_val_samples))
    disp_every = cfg.TEST.DISP_LOG_EVERY
    if DEBUG:
        num_val_samples = cfg.TEST.NUM_SAMPLES_DEBUGMODE
        disp_every = 1

    for index in range(num_val_samples):
        with torch.no_grad():


            # return image.copy(), semseg_label, np.array(image.shape), name
            image, label, _, _ = next(test_iter) # this is for dacs training # TODO


            if fixed_test_size:
                interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True) # height x widht
            else:
                interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)  # height x widht
            # forward pass
            # start_ts = time.time()

            # pred_main = model(image.cuda(device))  # this is for dacs training # TODO

            pred_main = model(image.cuda(device))['semantic']

            # logger.info('time taken: {}'.format( time.time() - start_ts))

            output = interp(pred_main).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
            label = label.numpy()[0]
            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
            if (index+1) % disp_every == 0:
                logger.info('{:d} / {:d}: {:0.2f}'.format(index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
    cIoU = per_class_iu(hist)
    mIoU = round(np.nanmean(cIoU) * 100, 2)

    name_classes = test_loader.dataset.class_names
    for ind_class in range(cfg.NUM_CLASSES):
        logger.info(name_classes[ind_class] + '\t' + str(round(cIoU[ind_class] * 100, 2)))
    logger.info('*** Current mIoU: {} ***'.format(mIoU))
    return cIoU, mIoU

