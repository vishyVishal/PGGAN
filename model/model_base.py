import torch
from torch import nn
import math
from torch.nn import functional as F
import numpy as np


class EqualizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(EqualizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.weight.data.normal_(0.0, 1.0)
        if self.bias is not None:
            self.bias.data.fill_(0)
        fan_in = kernel_size * kernel_size * in_channels
        self.scale = math.sqrt(2. / fan_in)

    def forward(self, x):
        return F.conv2d(input=x,
                        weight=self.weight.mul(self.scale),  # scale the weight on runtime
                        bias=self.bias,
                        stride=self.stride, padding=self.padding)


class EqualizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(EqualizedLinear, self).__init__(in_features, out_features, bias)
        self.weight.data.normal_(0.0, 1.0)
        if self.bias is not None:
            self.bias.data.fill_(0)
        fan_in = in_features
        self.scale = math.sqrt(2. / fan_in)

    def forward(self, x):
        return F.linear(x, weight=self.weight.mul(self.scale), bias=self.bias)


class GeneralizedDropout(nn.Module):
    def __init__(self, mode='mul', strength=0.4, axes=(0, 1), normalize=False):
        super(GeneralizedDropout, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode' % mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in
                     enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1

        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = torch.from_numpy(rnd).type(x.data.type())
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (
            self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str


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
