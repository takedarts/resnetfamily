#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer

def get_mnist():
  train_data, test_data = chainer.datasets.get_mnist()
  
  images, labels = train_data._datasets
  images = images.reshape(images.shape[0], 1, 28, 28)
  train_data._datasets = (images, labels)
    
  images, labels = test_data._datasets
  images = images.reshape(images.shape[0], 1, 28, 28)
  test_data._datasets = (images, labels)
    
  return train_data, test_data
