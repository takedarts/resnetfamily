#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import chainer

class CosineShift(chainer.training.extension.Extension):
  def __init__(self, attr, value, period, period_mult=1, optimizer=None):
    self._attr = attr
    self._value = value
    self._period = period
    self._period_mult = period_mult
    self._optimizer = optimizer

    if not hasattr(self._value, '__getitem__'):
      self._value = (self._value, 0)

  def initialize(self, trainer):
    self._update_value(trainer)

  def __call__(self, trainer):
    self._update_value(trainer)

  def _update_value(self, trainer):
    optimizer = self._optimizer or trainer.updater.get_optimizer('main')
    epoch = trainer.updater.epoch

    period_range = self._period
    period_start = 0
    period_end = period_range

    while period_end <= epoch:
      period_start = period_end
      period_range *= self._period_mult
      period_end += period_range

    n_max, n_min = self._value
    t_cur = epoch - period_start
    t_i = period_range
    value = n_min + 0.5 * (n_max - n_min) * (1 + math.cos((t_cur / t_i) * math.pi))

    setattr(optimizer, self._attr, value)

