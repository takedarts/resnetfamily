#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
ResNeXt Network (pre-activation)モジュール。
Residual Block : BN - Conv(1x1) - BN - ReLU - ConvGroup(3x3) - BN - ReLU - Conv(1x1) - BN
@author: Atsushi TAKEDA
'''
import chainer
import mylib

def reshape(x, channels):
  if x.shape[1] < channels:
    xp = chainer.cuda.get_array_module(x)
    p = xp.zeros((x.shape[0], channels - x.shape[1], x.shape[2], x.shape[3]), dtype=x.dtype)
    x = chainer.functions.concat((x, p), axis=1)
  elif x.shape[1] > channels:
    x = x[:, :channels, :]

  return x


class ResnextUnit(chainer.Chain):
  def __init__(self, in_channels, out_channels, units):
    super().__init__(norm0=chainer.links.BatchNormalization(in_channels),
                     conv1=chainer.links.Convolution2D(in_channels, out_channels, 1),
                     norm1=chainer.links.BatchNormalization(out_channels),
                     conv2=mylib.links.GroupedConvolution2D(out_channels, out_channels, units, 3, pad=1),
                     norm2=chainer.links.BatchNormalization(out_channels),
                     conv3=chainer.links.Convolution2D(out_channels, out_channels, 1),
                     norm3=chainer.links.BatchNormalization(out_channels))

  def __call__(self, x):
    x = self.norm0(x)
    x = self.conv1(x)
    x = self.norm1(x)
    x = chainer.functions.relu(x)
    x = self.conv2(x)
    x = self.norm2(x)
    x = chainer.functions.relu(x)
    x = self.conv3(x)
    x = self.norm3(x)

    return x


class ResnextBlock(chainer.ChainList):
  def __init__(self, in_channels, out_channels, units, depth):
    layers = [ResnextUnit(in_channels, out_channels, units)]
    layers += [ResnextUnit(out_channels, out_channels, units) for _ in range(depth - 1)]

    super().__init__(*layers)

  def __call__(self, x):
    for layer in self:
      y = layer(x)
      y += reshape(x, y.shape[1])
      x = y

    return x


class Network(chainer.Chain):
  def __init__(self, category, params):
    depth, width, units = params
    depth = (depth - 2) // 9
    width = width * units

    super().__init__(input=chainer.links.Convolution2D(None, width, 3, pad=1),
                     norm=chainer.links.BatchNormalization(width),
                     block1=ResnextBlock(width * 1, width * 1, units, depth),
                     block2=ResnextBlock(width * 1, width * 2, units, depth),
                     block3=ResnextBlock(width * 2, width * 4, units, depth),
                     output=chainer.links.Linear(width * 4, category))

  def __call__(self, x):
    x = self.input(x)
    x = self.norm(x)

    x = self.block1(x)

    x = chainer.functions.average_pooling_2d(x, 2)
    x = self.block2(x)

    x = chainer.functions.average_pooling_2d(x, 2)
    x = self.block3(x)

    x = chainer.functions.relu(x)
    x = chainer.functions.average_pooling_2d(x, x.shape[2])
    x = self.output(x)

    return x

