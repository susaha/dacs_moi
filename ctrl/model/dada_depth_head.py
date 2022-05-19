import torch.nn as nn
import torch
import logging



class DADADetphHead(nn.Module):
    def __init__(self):
        super(DADADetphHead, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info('ctrl/model/dada_depth_head.py --> class DADADetphHead -->  __init__()')

    def forward(self, x4_dec3):
        depth = torch.mean(x4_dec3, dim=1, keepdim=True)  # depth output
        return depth