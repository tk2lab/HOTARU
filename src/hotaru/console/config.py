import sys
import os

import tensorflow as tf

from .base import Command, _option


class ConfigCommand(Command):

    description = 'Update parameter'

    name = 'config'
    options = [
        _option('job-dir', 'j', 'target directory'),
        _option('name', None, ''),

        _option('imgs-file'),
        _option('mask-type'),

        _option('tau-rise', None, 'time constant of calicum raise (sec)'),
        _option('tau-fall', None, 'time constant of calcium fall (sec)'),
        _option('hz', None, 'sampling rate of data (1/sec)'),
        _option('tau-scale', None, ''),

        _option('gauss', 'g', 'size of gaussian filter (px)'),
        _option('radius-type', None, '{linear,log,manual}'),
        _option('radius', None, 'radius of cell (px)'),

        _option('thr-strength', None, ''),
        _option('thr-distance', None, ''),
        _option('shard', None, ''),

        _option('la', 'a', 'penalty coefficient of footprint'),
        _option('lu', 'u', 'penalty coefficient of spike'),
        _option('bx', 'x', 'penalty coefficient of spatical baseline'),
        _option('bt', 't', 'penalty coefficient of temporal baseline'),

        _option('batch', None, ''),
        _option('thr-firmness', None, ''),
        _option('learning-rate', 'l', ''),
        _option('step', None, ''),
        _option('epoch', None, ''),
        _option('tol', None, ''),
    ]

    def handle(self):
        self.set_job_dir()
        self._update_parameter('name', 'default', str)

        self._update_parameter('imgs-file', 'imgs.tif', str)
        self._update_parameter('mask-type', '0.pad', str)

        self._update_parameter('hz', 20.0)
        self._update_parameter('tau-rise', 0.08)
        self._update_parameter('tau-fall', 0.16)
        self._update_parameter('tau-scale', 6.0)

        self._update_parameter('gauss', 2.0)
        self._update_parameter('radius-type', 'log', str)
        self._update_parameter('radius', '2.0,100.0,16', float_tuple)

        self._update_parameter('thr-strength', 0.5)
        self._update_parameter('thr-distance', 1.5)
        self._update_parameter('shard', 1, int)

        self._update_parameter('la',  1.5)
        self._update_parameter('lu', 10.0)
        self._update_parameter('bx', 0.0)
        self._update_parameter('bt', 0.0)
        self._update_parameter('thr-firmness', 0.2)

        self._update_parameter('batch', 100, int)
        self._update_parameter('learning-rate', 0.01)
        self._update_parameter('step', 100, int)
        self._update_parameter('epoch', 100, int)
        self._update_parameter('tol', 1e-3)

        self.save_status()

        status = self.status.params
        for k, v in status.items():
            self.line('{}: {}'.format(k, v))

    def _update_parameter(self, name, default=None, dtype=float):
        status = self.status.params
        curr_val = status.get(name, None)
        val = self.option(name) or curr_val or default
        if val is None:
            self.line('missing: {name}', 'error')
            sys.exit(1)
        if val != curr_val:
            status[name] = dtype(val)


def float_tuple(val):
    return tuple(float(v) for v in val.split(','))
