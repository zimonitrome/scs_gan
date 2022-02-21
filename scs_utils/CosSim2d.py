# Copied from https://gist.github.com/ClashLuke/8f6521deef64789e76334f1b72a70d80
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class CosSim2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=None, dilation=1,
                 groups: int = 1, bias: bool = False, q_scale: float = 10, p_scale: float = 100):
        self.norm_pad = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        if padding is None:
            padding = self.norm_pad

        bias = False
        assert dilation == 1, "Dilation has to be 1 to use AvgPool2d as L2-Norm backend."
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert groups == in_channels or groups == 1, "Either depthwise or full convolution. Grouped not supported"
        super(CosSim2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.q_scale = q_scale
        self.p_scale = p_scale
        self.p = torch.nn.Parameter(torch.full((1,), p_scale ** 0.5))
        self.q = torch.nn.Parameter(torch.full((1,), q_scale ** 0.5))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        q = self.q / self.q_scale
        out = inp.square()
        if self.groups == 1:
            out = out.sum(1, keepdim=True)
        norm = F.avg_pool2d(out, self.kernel_size, 1, self.norm_pad) * np.prod(self.kernel_size)
        # norm = F.conv2d(inp, torch.ones_like(self.weight)[:1, :1], None, 1, self.norm_pad)
        norm = norm.sqrt() + q
        weight = self.weight / (self.weight.square().sum(dim=(1, 2, 3), keepdim=True).sqrt() + q)
        out = F.conv2d(inp / norm, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        abs = (out.square() + 1e-6).sqrt()
        sign = out / abs
        out = abs ** (self.p.square() / self.p_scale)
        out = out * sign
        return out