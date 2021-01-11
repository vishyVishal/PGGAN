import torch
from torch import nn
import math
from torch.nn.init import kaiming_normal_, calculate_gain


class WeightScaleLayer(nn.Module):
    def __init__(self, income_layer):
        super(WeightScaleLayer, self).__init__()
        self.scale = (torch.mean(income_layer.weight.data ** 2)) ** 0.5
        income_layer.weight.data.copy_(income_layer.weight.data / self.scale)
        self.bias = None
        if income_layer.bias is not None:
            self.bias = income_layer.bias
            income_layer.bias = None

    def forward(self, x):
        x = self.scale * x
        if self.bias is not None:
            x += self.bias[:, None, None]
        return x


class PixelNormLayer(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        channels = x.shape[1]
        return x / x.norm(dim=1, keepdim=True) * math.sqrt(channels)


class MinibatchStatConcatLayer(nn.Module):
    def __init__(self):
        super(MinibatchStatConcatLayer, self).__init__()
        self.adjust_std = torch.std

    def forward(self, x):
        shape = x.shape
        batch_std = self.adjust_std(x, dim=0, keepdim=True)
        vals = torch.mean(batch_std)
        vals = vals.repeat(shape[0], 1, shape[2], shape[3])
        return torch.cat([x, vals], dim=1)


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(-1, *self.shape)

    def extra_repr(self) -> str:
        return f'shape={self.shape}'


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def kaiming_init(layer, nonlinearity='Conv2d', param=None):
    if nonlinearity == 'leaky_relu':
        gain = calculate_gain('leaky_relu', param)
    else:
        gain = calculate_gain(nonlinearity)
    kaiming_normal_(layer.weight, a=gain)
