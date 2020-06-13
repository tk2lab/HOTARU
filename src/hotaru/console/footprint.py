from datetime import datetime
import os

import tensorflow as tf
import numpy as np

from .base import Command, option
from ..util.npy import save_numpy


class FootprintCommand(Command):

    description = 'Update footprint'

    name = 'footprint'
    options = [
        option('job-dir'),
        option('prev', flag=False, value_required=False),
        option('name', flag=False, default='default'),
        option('batch', flag=False, default=100),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        prev = self.option('prev')
        la = self.status['root']['la']
        lu = self.status['root']['lu']
        bx = self.status['root']['bx']
        bt = self.status['root']['bt']
        prev_key = self.status['spike_current']
        if self.option('prev'):
            prev_key = {v: k for k, v in self.status['spike'].items()}[prev]
        key = prev_key + (('footprint', la, lu, bx, bt),)
        self._handle('footprint', key)

    def create(self, key, stage):
        print('footprint')
        model = self.model
        model.spike.val = self.spike
        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'footprint',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
        model.update_footprint(
            batch=int(self.option('batch')),
            log_dir=log_dir,
            stage=stage,
        )
        return model.footprint.val

    def save(self, base, val):
        save_numpy(base, val)
        return val
