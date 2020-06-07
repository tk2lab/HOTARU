import os
import tensorflow as tf

from .base import Command, option


class ConfigCommand(Command):

    description = 'Update parameter'

    name = 'config'
    options = [
        option('job-dir', 'j', 'target directory', flag=False, value_required=False),
        option('hz', 'z', 'sampling rate of data (1/sec)', flag=False, value_required=False),
        option('tau-rise', 'r', 'time constant of calicum raise (sec)', flag=False, value_required=False),
        option('tau-fall', 'f', 'time constant of calcium fall (sec)', flag=False, value_required=False),
        option('tau-scale', 's', '', flag=False, value_required=False),
        option('gauss', 'g', 'size of gaussian filter (px)', flag=False, value_required=False),
        option('diameter', 'd', 'diameter of cell (px)', flag=False, multiple=True),
        option('la', 'a', 'penalty coefficient of footprint', flag=False, value_required=False),
        option('lu', 'u', 'penalty coefficient of spike', flag=False, value_required=False),
        option('bx', 'x', 'penalty coefficient of spatical baseline', flag=False, value_required=False),
        option('bt', 't', 'penalty coefficient of temporal baseline', flag=False, value_required=False),
    ]

    def handle(self):
        self.set_job_dir()
        self._update_parameter('hz', 20.0)
        self._update_parameter('tau-rise', 0.08)
        self._update_parameter('tau-fall', 0.16)
        self._update_parameter('tau-scale', 6.0)
        self._update_parameter('gauss', 2.0)
        self._update_parameter('diameter', [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self._update_parameter('la', 0.0)
        self._update_parameter('lu', 0.0)
        self._update_parameter('bx', 0.0)
        self._update_parameter('bt', 0.0)
        self.save_status()

    def _update_parameter(self, name, default=None):
        curr_val = self.status['root'].get(name, False)
        val = curr_val or self.option(name) or default
        if val is None:
            self.error('missing: {name}')
        elif val is not curr_val:
            if isinstance(val, list):
                self.status['root'][name] = tuple(float(v) for v in sorted(val))
            else:
                self.status['root'][name] = float(val)