import torch
from functools import partial
from torch import nn 
import torch.nn.functional as F 

# Inspired by: https://github.com/StephenHogg/SCS/blob/main/SCS/layer.py
class AbsPool(nn.Module):
    def __init__(self, pooling_module=None, *args, **kwargs):
        super(AbsPool, self).__init__()
        self.pooling_layer = pooling_module(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooling_layer(x.abs())

MaxAbsPool2d = partial(AbsPool, nn.MaxPool2d)

class AbsSplit(nn.Module):
    def forward(self, x):
        return torch.concat([F.relu(x), F.relu(-x)], dim=1)

class AbsSplitAlt(nn.Module):
    def forward(self, x):
        b,c,h,w = x.shape
        return torch.cat(
                (F.relu(x).unsqueeze(2), F.relu(-x).unsqueeze(2)),
                dim=2
            ).view(b,2*c,h,w)