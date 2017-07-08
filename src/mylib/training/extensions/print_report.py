#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from chainer.training import extensions
from chainer.training.extensions import log_report as log_report_module
from chainer.training.extensions import util

class PrintReport(extensions.PrintReport):
  def __call__(self, trainer):
    if isinstance(self._out, str):
      self._out = open(os.path.join(trainer.out, self._out), 'w')

    out = self._out

    if self._header:
      out.write(self._header)
      out.flush()
      self._header = None

    log_report = self._log_report

    if isinstance(log_report, str):
      log_report = trainer.get_extension(log_report)
    elif not isinstance(log_report, log_report_module.LogReport):
      raise TypeError('log report has a wrong type %s' % type(log_report))

    log = log_report.log
    log_len = self._log_len

    while len(log) > log_len:
      if out == sys.stdout:
        # delete the printed contents from the current cursor
        if os.name == 'nt':
          util.erase_console(0, 0)
        else:
          out.write('\033[J')

      self._print(log[log_len])
      out.flush()

      log_len += 1

    self._log_len = log_len
