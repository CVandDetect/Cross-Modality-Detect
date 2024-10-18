"""
在RT-DETR的基础上进行扩展，构建了一个能够同时输入红外和可见光图像的双模态融合目标检测网络FusionDETR
"""
import numpy as np
from src.core import register
from torch.nn import functional as F
import copy
import torch.nn as nn


__all__ = ['FusionDETR', ]

@register
class FusionDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone_rgb = copy.deepcopy(backbone)
        self.backbone_ir = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        rgb = x[:, 0:3, ...]
        ir = x[:, 3:, ...]
        new_ir = self.backbone_ir(ir)
        new_rgb = self.backbone_rgb(rgb)
        x = self.encoder(new_rgb, new_ir)
        x = self.decoder(x, targets)
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
