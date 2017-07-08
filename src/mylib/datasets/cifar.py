#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import chainer

class CifarDataset(chainer.dataset.DatasetMixin):
  def __init__(self, images, labels, pad=0, flip=False):
    self._pad = pad
    self._flip = flip

    if self._pad > 0:
      sides = ((0, 0), (0, 0), (self._pad, self._pad), (self._pad, self._pad))
      images = numpy.pad(images, sides, 'constant', constant_values=0)

    self._images = images
    self._labels = labels

  def __len__(self):
    return len(self._images)

  def get_example(self, i):
    image = self._images[i]
    label = self._labels[i]

    if self._pad > 0:
      r = numpy.random.randint(0, self._pad * 2 + 1, 2)
      s = (image.shape[1] - self._pad * 2, image.shape[2] - self._pad * 2)

      image = image[:, r[0]:r[0] + s[0], r[1]:r[1] + s[1]]

    if self._flip and numpy.random.randint(2) == 1:
      image = image[:, :, ::-1]

    return image, label


def get_cifar10():
  '''This function creates a cifar10 data set.
  All images in this data set are normalized.
  In addition, train images are augmented by random-padding-clipping and flipping.
  (This augmentation is standard of cifat10 benchmarks in 2015-2017)
  '''
  train, test = chainer.datasets.cifar.get_cifar10()

  train_dataset = CifarDataset(train._datasets[0], train._datasets[1], pad=4, flip=True)
  test_dataset = CifarDataset(test._datasets[0], test._datasets[1], pad=0, flip=False)

  return train_dataset, test_dataset


def get_cifar100():
  '''This function creates a cifar100 data set.
  All images in this data set are normalized.
  In addition, train images are augmented by random-padding-clipping and flipping.
  '''
  train, test = chainer.datasets.cifar.get_cifar100()

  train_dataset = CifarDataset(train._datasets[0], train._datasets[1], padding=4, flip=True)
  test_dataset = CifarDataset(test._datasets[0], test._datasets[1], padding=0, flip=False)

  return train_dataset, test_dataset

