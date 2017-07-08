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


class ResidualUnit(chainer.Chain):
  def __init__(self, in_channels, out_channels):
    super().__init__(norm0=chainer.links.BatchNormalization(in_channels),
                     conv1=chainer.links.Convolution2D(in_channels, out_channels, 3, pad=1),
                     norm1=chainer.links.BatchNormalization(out_channels),
                     conv2=chainer.links.Convolution2D(out_channels, out_channels, 3, pad=1),
                     norm2=chainer.links.BatchNormalization(out_channels))

  def __call__(self, x):
    x = self.norm0(x)
    x = self.conv1(x)
    x = self.norm1(x)
    x = chainer.functions.relu(x)
    x = self.conv2(x)
    x = self.norm2(x)

    return x


class ResidualBlock(chainer.ChainList):
  def __init__(self, in_channels, out_channels, depth):
    channels = [int((in_channels * (depth - i) + out_channels * i) / depth) for i in range(depth + 1)]

    super().__init__(*[ResidualUnit(channels[i], channels[i + 1]) for i in range(depth)])

  def __call__(self, x):
    for layer in self:
      y = layer(x)
      y += reshape(x, y.shape[1])
      x = y

    return x


class Network(chainer.Chain):
  def __init__(self, category, params):
    depth, alpha = params
    depth = (depth - 2) // 6

    super().__init__(input=chainer.links.Convolution2D(None, 16, 3, pad=1),
                     norm=chainer.links.BatchNormalization(16),
                     block1=ResidualBlock(16 + alpha * 0 // 3, 16 + alpha * 1 // 3, depth),
                     block2=ResidualBlock(16 + alpha * 1 // 3, 16 + alpha * 2 // 3, depth),
                     block3=ResidualBlock(16 + alpha * 2 // 3, 16 + alpha * 3 // 3, depth),
                     output=chainer.links.Linear(16 + alpha, category))

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

