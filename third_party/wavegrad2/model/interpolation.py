#This code is adopted from
#https://github.com/ivanvovk/WaveGrad
import torch

from third_party.wavegrad2.model.base import BaseModule


class InterpolationBlock(BaseModule):
    def __init__(self, scale_factor, mode='linear', align_corners=False, downsample=False):
        super(InterpolationBlock, self).__init__()
        self.downsample = downsample
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x):
        outputs = torch.nn.functional.interpolate(
            x,
            size=x.shape[-1] * self.scale_factor \
                if not self.downsample else x.shape[-1] // self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=False
        )
        return outputs
