#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import chainer

class IntervalTrigger(object):
  def __init__(self, period, unit):
    assert unit == 'epoch' or unit == 'iteration'

    self._period = period
    self._unit = unit
    self._count = 0

  def __call__(self, trainer):
    updater = trainer.updater

    if self._unit == 'epoch':
      count = updater.epoch // self._period

      if count != self._count:
        self._count = count
        return True
      else:
        return False

    else:
      return updater.iteration > 0 and updater.iteration % self._period == 0

  def serialize(self, serializer):
    if isinstance(serializer, chainer.serializer.Serializer):
      serializer('_count', json.dumps(self._count))
    else:
      self._count = json.loads(serializer('_count', ''))

