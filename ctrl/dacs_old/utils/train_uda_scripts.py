from ctrl.dacs_old.utils import transformsgpu
import torch
import numpy as np
from ctrl.utils.panoptic_deeplab.utils import get_module
from torchvision import transforms
import os
from ctrl.dacs_old.utils.helpers import colorize_mask
from ctrl.dacs.model.deeplabv2 import Res_Deeplab

class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean
    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

# save_image(cfg, inputs_u_s[i].cpu(), i_iter, current_epoch, 'input{}'.format(i+1), palette.CityScpates_palette, img_name_source, img_name_target)
def save_image(cfg, image, input_fname, pred_fname, palette):
    IMG_MEAN = np.array(cfg.TRAIN.IMG_MEAN, dtype=np.float32)
    save_path = cfg.TRAIN.DACS_VISUAL_RESULTS_DIR
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])
            image = restore_transform(image)
            #image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            fname_str = input_fname
            image.save(os.path.join(save_path, fname_str))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            fname_str = pred_fname
            colorized_mask.save(os.path.join(save_path, fname_str))


def update_ema_variables(cfg, ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    for ema_param, param in zip(get_module(ema_model, cfg.DISTRIBUTED).parameters(), get_module(model, cfg.DISTRIBUTED).parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def strongTransform(cfg, parameters, data=None, target=None):
    IMG_MEAN = np.array(cfg.TRAIN.IMG_MEAN, dtype=np.float32)
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.oneMix(mask=parameters["Mix"], data=data, target=target)
    data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean=torch.from_numpy(IMG_MEAN.copy()).cuda(), data=data, target=target)
    data, target = transformsgpu.gaussian_blur(blur=parameters["GaussianBlur"], data=data, target=target)
    data, target = transformsgpu.flip(flip=parameters["flip"], data=data, target=target)
    return data, target


def create_ema_model_dacs_ori(model):
    ema_model = Res_Deeplab(num_classes=16)
    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def create_ema_model(model, cfg):
    from ctrl.dacs_old.model.dacs_model import DACSModel
    ema_model = DACSModel(cfg)
    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def create_ema_model_dacs_panop(model, cfg):
    from ctrl.dacs_old.model.dacs_panop_model import DACSPanop
    ema_model = DACSPanop(cfg)
    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def create_ema_model_dacs_panop_v2_ablation(model, cfg):
    from ctrl.dacs_old.model.dacs_panop_model_v2_ablation import DACSPanopDF
    ema_model = DACSPanopDF(cfg)
    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def create_ema_model_dacs_panop_v2(model, cfg):
    from ctrl.dacs_old.model.dacs_panop_model_v2 import DACSPanopDF
    ema_model = DACSPanopDF(cfg)
    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def create_ema_model_dacs_panop_v3(model, cfg):
    from ctrl.dacs_old.model.dacs_panop_model_v3 import DACSPanopDF
    ema_model = DACSPanopDF(cfg)
    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def create_ema_model_dacs_panop_v4(model, cfg):
    from ctrl.model_panop.panoptic_deeplab import PanopticDeepLab
    ema_model = PanopticDeepLab(cfg)
    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def create_ema_model_dacs_panop_padnet(model, cfg):
    from ctrl.dacs_old.model.dacs_panop_model_padnet_v1 import DACSPanopPadNet
    ema_model = DACSPanopPadNet(cfg)
    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model