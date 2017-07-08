#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import chainer
import mylib
from chainer.training import extensions


def create_network(name, category, params):
  '''create the specified network model'''
  module = __import__('network.{0}'.format(name), fromlist=['Network'])
  cls = getattr(module, 'Network')

  return cls(category, params)


def load_dataset(name):
  '''load the specified datasets'''
  if name == 'mnist':
    return (10,) + mylib.datasets.get_mnist()
  elif name == 'cifar10':
    return (10,) + mylib.datasets.get_cifar10()
  elif name == 'cifar100':
    return (10,) + mylib.datasets.get_cifar100()


def main():
  parser = argparse.ArgumentParser(description='network trainer')
  parser.add_argument('dataset', metavar='DATASET', help='datasets name')
  parser.add_argument('network', metavar='NETWORK', help='network name')
  parser.add_argument('params', type=int, nargs='*', metavar='PARAMS', help='parameters')
  parser.add_argument('--learning', '-l', default='step', choices=('step', 'cosine', 'restart'),
                      metavar='NAME', help='name of learning rate control')
  parser.add_argument('--rate', '-r', type=float, default=0.1, 
                      metavar='LEARNING_RATE', help='initial leaning rate')
  parser.add_argument('--momentum', '-m', type=float, default=0.9, 
                      metavar='MOMENTUM', help='momentum of SGD')
  parser.add_argument('--decay', '-d', type=float, default=0.0001,
                      metavar='WEIGHT_DECAY', help='weight decay')
  parser.add_argument('--epoch', '-e', type=int, default=300,
                      metavar='EPOCH', help='number of epochs for training')
  parser.add_argument('--batchsize', '-b', type=int, default=128,
                      metavar='BATCH_SIZE', help='batch size of training')
  parser.add_argument('--procsize', '-p', type=int, default=None,
                      metavar='DATA_SIZE', help='number of images at a training process')
  parser.add_argument('--gpu', '-g', type=int, default=-1, 
                      metavar='GPU_ID', help='GPU ID')
  parser.add_argument('--no-check', action='store_true', default=False, help='without type check of variables')
  args = parser.parse_args()

  if args.procsize is None:
    args.procsize = args.batchsize

  if args.no_check:
    chainer.config.type_check = False

  name = '{}-{}-{}-{}'.format(args.dataset, args.network, '-'.join([str(v) for v in args.params]), args.learning)
  base_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
  result_dir = os.path.normpath(os.path.join(base_dir, 'result', name))

  # load data-set
  category, train_data, test_data = load_dataset(args.dataset)

  # create a neural network
  network = create_network(args.network, category, args.params)
  lossfun = chainer.functions.softmax_cross_entropy
  accfun = chainer.functions.accuracy
  classifier = chainer.links.Classifier(network, lossfun=lossfun, accfun=accfun)

  if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    classifier.to_gpu()

  # create optimizer
  optimizer = chainer.optimizers.MomentumSGD(lr=args.rate, momentum=args.momentum)
  optimizer.setup(classifier)
  optimizer.add_hook(chainer.optimizer.WeightDecay(args.decay))

  # create data iterators
  train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize, repeat=True, shuffle=True)
  test_iter = chainer.iterators.SerialIterator(test_data, args.procsize, repeat=False, shuffle=False)

  # create trainer
  updater = mylib.training.StandardUpdater(train_iter, optimizer, device=args.gpu, procsize=args.procsize)
  trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)

  # extension for evaluation
  trainer.extend(extensions.Evaluator(test_iter, classifier, device=args.gpu))

  # extension for controlling learning rate
  if args.learning == 'step':
    trainer.extend(mylib.training.extensions.StepShift('lr', args.rate, args.epoch))
  elif args.learning == 'cosine':
    trainer.extend(mylib.training.extensions.CosineShift('lr', args.rate, args.epoch, 1))
  elif args.learning == 'restart':
    trainer.extend(mylib.training.extensions.CosineShift('lr', args.rate, 10, 2))

  # extensions for logging
  plot_err_keys = ['main/loss', 'validation/main/loss']
  plot_acc_keys = ['main/accuracy', 'validation/main/accuracy']
  print_keys = ['epoch',
                'main/loss', 'validation/main/loss',
                'main/accuracy', 'validation/main/accuracy',
                'elapsed_time']
  trigger = mylib.training.trigger.IntervalTrigger

  trainer.extend(mylib.training.extensions.dump_graph('main/loss', out_name="variable.dot", remove_variable=False))
  trainer.extend(mylib.training.extensions.dump_graph('main/loss', out_name="function.dot", remove_variable=True))
  trainer.extend(mylib.training.extensions.dump_network_size(filename='size.txt'))

  trainer.extend(extensions.snapshot(filename='snapshot.npz'), trigger=trigger(1, 'epoch'))
  trainer.extend(mylib.training.extensions.Bestshot(filename='bestshot.npz', trigger=trigger(1, 'epoch')))

  trainer.extend(mylib.training.extensions.LogReport(log_name='log.txt', trigger=trigger(1, 'epoch')))
  trainer.extend(mylib.training.extensions.PrintReport(print_keys, log_report='LogReport'))
  trainer.extend(mylib.training.extensions.PrintReport(print_keys, log_report='LogReport', out='out.txt'))

  trainer.extend(mylib.training.extensions.PlotReport(plot_err_keys, 'epoch', file_name='loss.png',
                                                      marker=None, trigger=trigger(1, 'epoch')))
  trainer.extend(mylib.training.extensions.PlotReport(plot_acc_keys, 'epoch', file_name='accuracy.png',
                                                      marker=None, trigger=trigger(1, 'epoch')))

  # resume setting
  snapshot = os.path.join(result_dir, 'snapshot.npz')

  if os.path.isfile(snapshot):
    chainer.serializers.load_npz(snapshot, trainer)

  # start
  trainer.run()


if __name__ == '__main__':
  main()

