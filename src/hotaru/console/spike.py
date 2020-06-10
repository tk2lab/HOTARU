from datetime import datetime
import os

import tensorflow as tf
import numpy as np

from .base import Command, option


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
        prev_key = self.status['footprint_current']
        if self.option('prev'):
            prev_key = {v: k for k, v in self.status['footprint'].items()}[self.option('prev')]
        self.key = prev_key + ('spike', (tau1, tau2, hz, tauscale, la, lu, bx, bt))
        self._handle('spike')

    def create(self):
        self.line('spike')
        model = self.model
        model.footprint.val = self.footprint
        log_dir = os.path.join(self.application.job_dir, 'logs', 'spike', datetime.now().strftime('%Y%m%d-%H%M%S'))
        model.update_spike(
            batch=int(self.option('batch')),
            log_dir=log_dir,
        )
        return model.spike.val

    def save(self, base, val):
        np.save(base + '.npy', val)
