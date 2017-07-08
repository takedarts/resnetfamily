#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer

class StepShift(chainer.training.extension.Extension):
  def __init__(self, attr, value, period, optimizer=None):
    self._attr = attr
    self._value = value
    self._period = period
    self._optimizer = optimizer

  def initialize(self, trainer):
    self._update_value(trainer)

  def __call__(self, trainer):
    self._update_value(trainer)

  def _update_value(self, trainer):
    optimizer = self._optimizer or trainer.updater.get_optimizer('main')
    current = self._period - trainer.updater.epoch

    if current <= self._period // 4:
      value = self._value * 0.01
    elif current <= self._period // 2:
      value = self._value * 0.1
    else:
      value = self._value

    setattr(optimizer, self._attr, value)

