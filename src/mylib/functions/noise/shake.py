#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import cuda
from chainer import configuration
from chainer import function
from chainer.utils import type_check

class ShakeNoiseFunction(function.Function):
  def check_type_forward(self, in_types):
    type_check.expect(in_types.size() == 2)
    type_check.expect(in_types[0].dtype.kind == 'f')
    type_check.expect(in_types[1].dtype.kind == 'f')

  def forward(self, inputs):
    xp = cuda.get_array_module(*inputs)
    x1, x2 = inputs

    mask1 = xp.random.randint(0, 2, x1.shape[0]).astype(xp.float32)
    mask2 = 1 - mask1

    x1 = x1 * mask1[:, xp.newaxis, xp.newaxis, xp.newaxis]
    x2 = x2 * mask2[:, xp.newaxis, xp.newaxis, xp.newaxis]

    return x1 + x2,

  def backward(self, inputs, grad):
    xp = cuda.get_array_module(*inputs)
    g = grad[0]

    mask1 = xp.random.randint(0, 2, g.shape[0]).astype(xp.float32)
    mask2 = 1 - mask1

    g1 = g * mask1[:, xp.newaxis, xp.newaxis, xp.newaxis]
    g2 = g * mask2[:, xp.newaxis, xp.newaxis, xp.newaxis]

    return g1, g2


def shake_noise(x1, x2):
  if configuration.config.train:
    return ShakeNoiseFunction()(x1, x2)
  else:
    return (x1 + x2) / 2
