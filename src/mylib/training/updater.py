#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import chainer
from chainer.dataset import convert

class StandardUpdater(chainer.training.StandardUpdater):
  def __init__(self, iterator, optimizer, converter=convert.concat_examples,
               device=None, loss_func=None, procsize=None):
    super().__init__(iterator, optimizer, converter, device, loss_func)
    self.procsize = procsize
  
  def update_core(self):
    batch = self.get_iterator('main').next()
    optimizer = self.get_optimizer('main')
    loss_func = self.loss_func or optimizer.target 
    
    repeats = max(math.ceil(len(batch) / self.procsize), 1)
    
    optimizer.target.cleargrads()
    
    for i in range(repeats):
      start = len(batch) * i // repeats 
      end = len(batch) * (i + 1) // repeats 
      in_arrays = self.converter(batch[start:end], self.device)
    
      with chainer.function.force_backprop_mode():
        if isinstance(in_arrays, tuple):
          loss = loss_func(*in_arrays)
        elif isinstance(in_arrays, dict):
          loss = loss_func(**in_arrays)
        else:
          loss = loss_func(in_arrays)
          
      loss.backward()
    
    if repeats != 1:
      for _, v in optimizer.target.namedparams():
        grad = v.grad
        
        if grad is not None:
          grad /= repeats
          v.grad = grad
      
    optimizer.update()
      
