#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Atsushi TAKEDA
'''

import os
import re
import json
import matplotlib.pyplot

GRAPH_FIGURE_SIZE = (8, 5)
GRAPH_FONT_SIZE = 12

class Data(object):
  def __init__(self, path):
    self.path = path
    self._read_size()
    self._read_log()

  def _read_size(self):
    with open(os.path.join(self.path, 'size.txt'), 'r') as handle:
      for line in handle:
        if line[:3] == '/: ':
          self.size = int(line[3:])
          return

    raise Exception('size data is not found')

  def _read_log(self):
    with open(os.path.join(self.path, 'log.txt'), 'r') as handle:
      self.log = json.load(handle)

    for v in self.log:
      v['main/error'] = 1.0 - v['main/accuracy']
      v['validation/main/error'] = 1.0 - v['validation/main/accuracy']

      if v['main/loss'] >= 1000:
        v['main/loss'] = float('nan')

      if v['validation/main/loss'] >= 1000:
        v['validation/main/loss'] = float('nan')

  def make_graph(self, path):
    self._make_graph([('main/loss', 'train loss'), ('validation/main/loss', 'test loss')],
                     'loss', '{}_loss.png'.format(path))
    self._make_graph([('main/error', 'train error'), ('validation/main/error', 'test error')],
                     'error rate', '{}_error.png'.format(path))

  def _make_graph(self, keys, label, file):
    figure = matplotlib.pyplot.figure(figsize=GRAPH_FIGURE_SIZE)
    plot = figure.add_subplot(1, 1, 1)

    x = [v['epoch'] for v in self.log]

    y_max = 0.001
    x_max = max(x)

    for key, name in keys:
      y = [v[key] for v in self.log]
      y_max = max(max(y) * 1.1, y_max)

      plot.plot(x, y, label=name, linewidth=1)

    plot.set_xlim(0, x_max)
    plot.set_ylim(0, y_max)
    plot.set_xlabel('epoch', fontsize=GRAPH_FONT_SIZE)
    plot.set_ylabel(label, fontsize=GRAPH_FONT_SIZE)
    plot.tick_params(labelsize=GRAPH_FONT_SIZE)
    plot.legend(loc='upper right', fontsize=GRAPH_FONT_SIZE)

    figure.savefig(file, bbox_inches='tight')
    matplotlib.pyplot.close()


class Result(object):
  def __init__(self, path):
    self.path = path
    self.data_list = []

    for meta in self._read_meta():
      data = Data(os.path.join(self.path, meta[0]))
      data.flag = meta[1]
      data.name = meta[2]

      self.data_list.append(data)

  def _read_meta(self):
    meta_list = []
    paths = []

    regex = re.compile(r'^([^:]+):([^\/]+)\/(.*)$')

    if os.path.isfile(os.path.join(self.path, 'meta.txt')):
      with open(os.path.join(self.path, 'meta.txt'), 'r') as handle:
        for line in handle:
          m = regex.match(line)

          if not m:
            continue

          path = m.group(1).strip()
          flag = m.group(2).strip().lower() == 'true'
          name = m.group(3).strip()

          if not os.path.isdir(os.path.join(self.path, path)):
            continue

          meta_list.append((path, flag, name))
          paths.append(path)

    for item in os.listdir(self.path):
      if not os.path.isdir(os.path.join(self.path, item)):
        continue

      if not os.path.isfile(os.path.join(self.path, item, 'log.txt')):
        continue

      if item in paths:
        continue

      meta_list.append((item, False, item))

    with open(os.path.join(self.path, 'meta.txt'), 'w') as handle:
      for meta in meta_list:
        handle.write("{}: {}/{}\n".format(*meta))

    return meta_list

  def make_graph(self):
    for data in self.data_list:
      if data.flag:
        data.make_graph(os.path.join(self.path, os.path.basename(data.path)))

    self._make_graph('main/loss', 'loss (train)', 'train_loss.png')
    self._make_graph('main/error', 'error rate (train)', 'train_error.png')
    self._make_graph('validation/main/loss', 'loss (validation)', 'test_loss.png')
    self._make_graph('validation/main/error', 'error rate (validation)', 'test_error1.png')
    self._make_graph('validation/main/error', 'error rate (validation)', 'test_error2.png', yrange='min')

  def _make_graph(self, key, name, file, yrange='max'):
    figure = matplotlib.pyplot.figure(figsize=GRAPH_FIGURE_SIZE)
    plot = figure.add_subplot(1, 1, 1)

    y_max = 0.001
    x_max = 0

    for data in self.data_list:
      if not data.flag:
        continue

      x = [v['epoch'] for v in data.log]
      y = [v[key] for v in data.log]

      x_max = max(max(x), x_max)

      if yrange == 'min':
        y_max = max(min(y) * 2.5, y_max)
      else:
        y_max = max(max(y) * 1.1, y_max)

      plot.plot(x, y, label=data.name, linewidth=1)

    plot.set_xlim(0, x_max)
    plot.set_ylim(0, y_max)
    plot.set_xlabel('epoch', fontsize=GRAPH_FONT_SIZE)
    plot.set_ylabel(name, fontsize=GRAPH_FONT_SIZE)
    plot.tick_params(labelsize=GRAPH_FONT_SIZE)
    plot.legend(loc='upper right', fontsize=GRAPH_FONT_SIZE)

    figure.savefig(os.path.join(self.path, file), bbox_inches='tight')
    matplotlib.pyplot.close()

  def __str__(self):
    text = [['name', 'size', 'loss(train)', 'error(train)', 'loss(test)', 'error(test)']]

    for data in self.data_list:
      name = data.name
      size = '{:.2f}m'.format(data.size / 1000000)
      train_loss = '{:.4f}'.format(min([v['main/loss'] for v in data.log]))
      train_error = '{:.4f}'.format(min([v['main/error'] for v in data.log]))
      test_loss = '{:.4f}'.format(min([v['validation/main/loss'] for v in data.log]))
      test_error = '{:.4f}'.format(min([v['validation/main/error'] for v in data.log]))

      text.append([name, size, train_loss, train_error, test_loss, test_error])

    spaces = [0] * len(text[0])

    for line in text:
      spaces = [max(spaces[i], len(v)) for i, v in enumerate(line)]

    fmt = ', '.join(['{{:{}{}}}'.format('' if i == 0 else '>', v) for i, v in enumerate(spaces)])
    text = [fmt.format(*v) for v in text]

    return "\n".join(text)


def main():
  basedir = os.path.join(os.path.dirname(__file__), os.path.pardir)
  result = Result(os.path.normpath(os.path.join(basedir, 'result')))

  result.make_graph()
  print(result)


if __name__ == '__main__':
  main()
