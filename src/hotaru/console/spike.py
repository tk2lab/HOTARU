import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base import Command, option
from ..util.npy import save_numpy


class SpikeCommand(Command):

    description = 'Update spike'

    name = 'spike'
    options = [
        option('job-dir'),
        option('prev', flag=False, value_required=False),
        option('name', flag=False, default='default'),
        option('batch', flag=False, default=100),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        tau1 = self.status['root']['tau-fall']
        tau2 = self.status['root']['tau-rise']
        hz = self.status['root']['hz']
        tauscale = self.status['root']['tau-scale']
        la = self.status['root']['la']
        lu = self.status['root']['lu']
        bx = self.status['root']['bx']
        bt = self.status['root']['bt']
        prev_key = self.status['clean_current']
        if self.option('prev'):
            prev_key = {v: k for k, v in self.status['clean'].items()}[self.option('prev')]
        key = prev_key + (('spike', tau1, tau2, hz, tauscale, la, lu, bx, bt),)
        self._handle('spike', key)

    def create(self, key, stage):
        self.line('spike')
        model = self.model
        clean = self.clean
        score = clean.max(axis=1)
        clean /= score[:, None]
        cond = score > self.status['root']['thr-firmness']
        clean = clean[cond]
        model.footprint.val = clean
        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'spike',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
        model.update_spike(
            batch=int(self.option('batch')),
            log_dir=log_dir,
            stage=stage,
        )
        return model.spike.val

    def save(self, base, val):
        save_numpy(base, val)
        return val
