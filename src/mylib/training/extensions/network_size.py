#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import os
import chainer

def dump_network_size(filename='network_size.txt'):
  def trigger(trainer):
    return trainer.updater.iteration == 1

  @chainer.training.extension.make_extension(trigger=trigger)
  def dump_network_size(trainer):
    model = trainer.updater.get_optimizer('main').target
    network_size = {}

    for n, v in model.namedparams():
      size = functools.reduce(lambda x, y: x * y, v.shape)
      names = n.split('/')
      names = ['/'] + ['/'.join(names[:i + 1]) for i in range(1, len(names))]

      for name in names:
        if name not in network_size:
          network_size[name] = 0

        network_size[name] += size

    network_size = [(n, s) for n, s in network_size.items()]
    network_size.sort(key=lambda x: x[0])

    path = os.path.join(trainer.out, filename)

    with open(path, 'w') as handle:
      for v in network_size:
        handle.write("{}: {}\n".format(*v))

  return dump_network_size

