import sys
import os

import tensorflow as tf

from .base import Command, _option


class ConfigCommand(Command):

    description = 'Update parameter'

    name = 'config'
    options = [
        _option('job-dir', 'j', 'target directory'),
        _option('imgs-file'),
        _option('mask-type'),
        _option('hz', None, 'sampling rate of data (1/sec)'),
        _option('name', None, ''),
        _option('tau-rise', None, 'time constant of calicum raise (sec)'),
        _option('tau-fall', None, 'time constant of calcium fall (sec)'),
        _option('tau-scale', None, ''),
        _option('gauss', 'g', 'size of gaussian filter (px)'),
        _option('radius-min', None, 'radius of cell (px)'),
        _option('radius-max', None, 'radius of cell (px)'),
        _option('radius-num', None, 'radius of cell (px)'),
        _option('la', 'a', 'penalty coefficient of footprint'),
        _option('lu', 'u', 'penalty coefficient of spike'),
        _option('bx', 'x', 'penalty coefficient of spatical baseline'),
        _option('bt', 't', 'penalty coefficient of temporal baseline'),
        _option('shard', None, ''),
        _option('skip', None, ''),
        _option('thr-gl', None, ''),
        _option('thr-dist', None, ''),
        _option('thr-firmness', None, ''),
        _option('learning-rate', 'l', ''),
        _option('batch', None, ''),
        _option('step', None, ''),
        _option('epoch', None, ''),
        _option('tol', None, ''),
    ]

    def handle(self):
        self.set_job_dir()
        self._assign_parameter('imgs-file', 'imgs.tif')
        self._assign_parameter('mask-type', '0.pad')
        self._update_parameter('hz', 20.0)
        self._update_parameter('name', 'default', str)
        self._update_parameter('gauss', 2.0)
        self._update_parameter('radius-min',  3.0)
        self._update_parameter('radius-max', 11.0)
        self._update_parameter('radius-num', 10, int)
        self._update_parameter('tau-rise', 0.08)
        self._update_parameter('tau-fall', 0.16)
        self._update_parameter('tau-scale', 6.0)
        self._update_parameter('la', 1.5)
        self._update_parameter('lu', 5.0)
        self._update_parameter('bx', 0.0)
        self._update_parameter('bt', 0.0)
        self._update_parameter('shard', 1, int)
        self._update_parameter('thr-gl', 0.5)
        self._update_parameter('thr-dist', 1.5)
        self._update_parameter('thr-firmness', 0.1)
        self._update_parameter('learning-rate', 0.01)
        self._update_parameter('batch', 100, int)
        self._update_parameter('step', 100, int)
        self._update_parameter('epoch', 100, int)
        self._update_parameter('tol', 1e-3)
        self.save_status()

    def _assign_parameter(self, name, default_val=None, dtype=str):
        status = self.status['root']
        curr_val = status.get(name, None)
        val = self.option(name) or curr_val or default_val
        if val is None:
            self.line(f'missing: {name}', 'error')
            sys.exit(1)
        if val != curr_val:
            mask_file = os.path.join(self.work_dir, 'mask.npy')
            if tf.io.gfile.exists(mask_file):
                self.line(f'config mismatch: {name}', 'error')
                sys.exit(1)
            status[name] = dtype(val)

    def _update_parameter(self, name, default=None, dtype=float):
        status = self.status['root']
        curr_val = status.get(name, None)
        val = self.option(name) or curr_val or default
        if val is None:
            self.line('missing: {name}', 'error')
            sys.exit(1)
        if val != curr_val:
            status[name] = dtype(val)
