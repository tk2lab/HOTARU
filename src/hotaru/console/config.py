import os

import tensorflow as tf

from .base import Command, option


class ConfigCommand(Command):

    description = 'Update parameter'

    name = 'config'
    options = [
        option('job-dir', 'j', 'target directory', flag=False, value_required=False),
        option('imgs-file', flag=False, value_required=False),
        option('mask-type', flag=False, value_required=False),
        option('hz', None, 'sampling rate of data (1/sec)', flag=False, value_required=False),
        option('name', None, '', flag=False, default='init'),
        option('tau-rise', None, 'time constant of calicum raise (sec)', flag=False, value_required=False),
        option('tau-fall', None, 'time constant of calcium fall (sec)', flag=False, value_required=False),
        option('tau-scale', None, '', flag=False, value_required=False),
        option('gauss', 'g', 'size of gaussian filter (px)', flag=False, value_required=False),
        option('radius', 'r', 'radius of cell (px)', flag=False, multiple=True),
        option('la', 'a', 'penalty coefficient of footprint', flag=False, value_required=False),
        option('lu', 'u', 'penalty coefficient of spike', flag=False, value_required=False),
        option('bx', 'x', 'penalty coefficient of spatical baseline', flag=False, value_required=False),
        option('bt', 't', 'penalty coefficient of temporal baseline', flag=False, value_required=False),
        option('shard', None, '', flag=False, value_required=False),
        option('skip', None, '', flag=False, value_required=False),
        option('thr-gl', None, '', flag=False, value_required=False),
        option('thr-dist', None, '', flag=False, value_required=False),
        option('thr-firmness', None, '', flag=False, value_required=False),
        option('learning-rate', 'l', '', flag=False, value_required=False),
        option('batch', None, '', flag=False, value_required=False),
        option('step', None, '', flag=False, value_required=False),
        option('epoch', None, '', flag=False, value_required=False),
        option('tol', None, '', flag=False, value_required=False),
    ]

    def handle(self):
        self.set_job_dir()
        self._assign_parameter('imgs-file', 'imgs.tif')
        self._assign_parameter('mask-type', '0.pad')
        self._update_parameter('hz', 20.0)
        self._update_parameter('name', 'default', str)
        self._update_parameter('gauss', 2.0)
        self._update_parameter('radius', [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self._update_parameter('tau-rise', 0.08)
        self._update_parameter('tau-fall', 0.16)
        self._update_parameter('tau-scale', 6.0)
        self._update_parameter('la', 1.5)
        self._update_parameter('lu', 5.0)
        self._update_parameter('bx', 0.0)
        self._update_parameter('bt', 0.0)
        self._update_parameter('shard', 1, int)
        self._update_parameter('thr-gl', 0.7)
        self._update_parameter('thr-dist', 1.6)
        self._update_parameter('thr-firmness', 0.1)
        self._update_parameter('learning-rate', 0.01)
        self._update_parameter('batch', 100, int)
        self._update_parameter('step', 100, int)
        self._update_parameter('epoch', 100, int)
        self._update_parameter('tol', 1e-3)
        self.save_status()

    def _assign_parameter(self, name, default_val=None, dtype=str):
        current_val = self.status['root'].get(name, None)
        val = self.option(name)
        if current_val is None:
            val = val or default_val
            if val is None:
                raise RuntimeError(f'config missing: {name}')
            if isinstance(val, list):
                val = tuple(dtype(v) for v in val)
            else:
                val = dtype(val)
            self.status['root'][name] = val
        elif val and val != current_val:
            raise RuntimeError(f'config mismatch: {name}')
        return self.status['root'][name]

    def _update_parameter(self, name, default=None, dtype=float):
        status = self.status['root']
        curr_val = status.get(name, None)
        val = self.option(name) or curr_val or default
        if val is None:
            self.error('missing: {name}')
        elif val != curr_val:
            if isinstance(val, list):
                status[name] = tuple(sorted(dtype(v) for v in val))
            else:
                status[name] = dtype(val)
