#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import shutil
import tempfile
import chainer

from chainer.serializers import npz

class Bestshot(chainer.training.extension.Extension):
  def __init__(self, savefun=npz.save_npz, filename='bestshot.npz', trigger=(1, 'epoch'),
               key='validation/main/accuracy', comp='max'):
    self._savefun = savefun
    self._filename = filename
    self._trigger = chainer.training.trigger.get_trigger(trigger)
    self._key = key
    self._comp = comp
    self._value = None

    if self._comp == 'min':
      self._comp = lambda x, y: y - x
    elif self._comp == 'max':
      self._comp = lambda x, y: x - y

    self._init_summary()

  def __call__(self, trainer):
    if self._key in trainer.observation:
      self._summary.add({self._key: trainer.observation[self._key]})

    if self._trigger(trainer):
      stats = self._summary.compute_mean()
      value = float(stats[self._key])

      if self._value is None or self._comp(value, self._value) > 0:
        self._value = value
        self._save(trainer)

      self._init_summary()

  def _save(self, trainer):
    filename = self._filename.format(trainer)
    prefix = 'tmp' + filename

    fd, tmppath = tempfile.mkstemp(prefix=prefix, dir=trainer.out)

    try:
        self._savefun(tmppath, trainer)
    except Exception:
        os.close(fd)
        os.remove(tmppath)
        raise

    os.close(fd)
    shutil.move(tmppath, os.path.join(trainer.out, filename))

  def serialize(self, serializer):
    self._value = json.loads(serializer('value', json.dumps(self._value)))
    self._trigger.serialize(serializer['trigger'])

  def _init_summary(self):
    self._summary = chainer.reporter.DictSummary()
