#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer.training import extensions

class PlotReport(extensions.PlotReport):
  def serialize(self, serializer):
    super().serialize(serializer)
    self._trigger.serialize(serializer['_trigger'])
