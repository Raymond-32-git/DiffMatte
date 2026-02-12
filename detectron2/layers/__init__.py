import torch
import torch.nn as nn
from torch.nn import functional as F

class ShapeSpec:
    def __init__(self, channels=None, height=None, width=None, stride=None):
        self.channels = channels
        self.height = height
        self.width = width
        self.stride = stride

class CNNBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

def Conv2d(*args, **kwargs):
    norm = kwargs.pop("norm", None)
    activation = kwargs.pop("activation", None)
    m = nn.Conv2d(*args, **kwargs)
    return m

class MockLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x):
        if x.ndim == 4:
            # NCHW normalization
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            w = self.weight
            b = self.bias
            # Broadcast weight/bias across spatial dims
            if w.ndim == 1:
                w = w[:, None, None]
            if b.ndim == 1:
                b = b[:, None, None]
            return w * x + b
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

def get_norm(norm, out_channels):
    if not norm: return None
    if norm == "BN": return nn.BatchNorm2d(out_channels)
    if norm == "GN": return nn.GroupNorm(32, out_channels)
    if norm == "LN": return MockLayerNorm(out_channels)
    return None
