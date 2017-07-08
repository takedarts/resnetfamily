#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Residual Network (pre-activation)モジュール。
Residual Block : BN - Conv(3x3) - BN - ReLU - Conv(3x3) - BN
@author: Atsushi TAKEDA
'''
import chainer


def reshape(x, channels):
  if x.shape[1] < channels:
    xp = chainer.cuda.get_array_module(x)
    p = xp.zeros((x.shape[0], channels - x.shape[1], x.shape[2], x.shape[3]), dtype=x.dtype)
    x = chainer.functions.concat((x, p), axis=1)
  elif x.shape[1] > channels:
    x = x[:, :channels, :]

  return x


class DenseUnit(chainer.Chain):
  def __init__(self, in_channels, out_channels):
    super().__init__(norm0=chainer.links.BatchNormalization(in_channels),
                     conv1=chainer.links.Convolution2D(in_channels, out_channels, 3, pad=1),
                     norm1=chainer.links.BatchNormalization(out_channels))

  def __call__(self, x):
    x = self.norm0(x)
    x = chainer.functions.relu(x)
    x = self.conv1(x)
    x = self.norm1(x)

    return x


class DenseBlock(chainer.ChainList):
  def __init__(self, in_channels, growth, depth):
    units = [DenseUnit(in_channels + growth * i, growth) for i in range(depth)]

    super().__init__(*units)

  def __call__(self, x):
    for layer in self:
      y = layer(x)
      x = chainer.functions.concat((x, y), axis=1)

    return x


class Network(chainer.Chain):
  def __init__(self, category, params):
    depth, growth = params
    depth = (depth - 2) // 3

    super().__init__(input=chainer.links.Convolution2D(None, 16, 3, pad=1),
                     norm=chainer.links.BatchNormalization(16),
                     block1=DenseBlock(16 + growth * depth * 0, growth, depth),
                     conv1=chainer.links.Convolution2D(16 + growth * depth * 1, 16 + growth * depth * 1, 1),
                     block2=DenseBlock(16 + growth * depth * 1, growth, depth),
                     conv2=chainer.links.Convolution2D(16 + growth * depth * 2, 16 + growth * depth * 2, 1),
                     block3=DenseBlock(16 + growth * depth * 2, growth, depth),
                     output=chainer.links.Linear(16 + growth * depth * 3, category))

  def __call__(self, x):
    x = self.input(x)
    x = self.norm(x)

    x = self.block1(x)
    x = self.conv1(x)

    x = chainer.functions.average_pooling_2d(x, 2)
    x = self.block2(x)
    x = self.conv2(x)

    x = chainer.functions.average_pooling_2d(x, 2)
    x = self.block3(x)

    x = chainer.functions.relu(x)
    x = chainer.functions.average_pooling_2d(x, x.shape[2])
    x = self.output(x)

    return x

