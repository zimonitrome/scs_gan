# Copied from https://gist.github.com/ClashLuke/8f6521deef64789e76334f1b72a70d80
import torch
from torch import nn
from torch.nn import functional as F


class CosSimConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=None, dilation=1,
                 groups: int = 1, bias: bool = False, p = True, q_scale: float = 10, p_scale: float = 100):
        if padding is None:
            if int(torch.__version__.split('.')[1]) >= 10:
                padding = "same"
            else:
                padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2  # This doesn't support even kernels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        assert dilation == 1, "Dilation has to be 1 to use AvgPool2d as L2-Norm backend."
        assert groups == in_channels or groups == 1, "Either depthwise or full convolution. Grouped not supported"
        super(CosSimConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)

        self.q_scale = q_scale  # q scale is missing at normalization of input. minor difference, but necessary
        self.q = torch.nn.Parameter(torch.full((1,), q_scale ** 0.5))

        # For "true" SCS:
        if p:
            self.p_scale = p_scale
            self.p = torch.nn.Parameter(torch.full((out_channels,), 2 ** 0.5 + p_scale))
        else:
            self.p = None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp.square()
        if self.groups == 1:
            out = out.sum(1, keepdim=True)
        norm = F.conv2d(out, torch.ones_like(self.weight[:1, :1]), None, self.stride, self.padding, self.dilation) + 1e-6

        q = self.q.square() / self.q_scale
        weight = self.weight / (self.weight.square().sum(dim=(1, 2, 3), keepdim=True).sqrt() + q)
        out = F.conv2d(inp, weight, self.bias, self.stride, self.padding, self.dilation, self.groups) / norm.sqrt()

        if self.p is None:
            return out

        # For "true" SCS (it's ~200x slower):
        sign = torch.sign(out)

        out = torch.abs(out) + 1e-6
        p_sqr = (self.p / self.p_scale) ** 2
        out = out.pow(p_sqr.reshape(1, -1, 1, 1))
        return sign * out