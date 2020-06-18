from datetime import datetime
import os

import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..util.npy import save_numpy


class FootprintCommand(Command):

    description = 'Update footprint'

    name = 'footprint'
    options = [
        _option('job-dir'),
        _option('prev'),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        prev = self.option('prev')
        la = self.status['root']['la']
        lu = self.status['root']['lu']
        bx = self.status['root']['bx']
        bt = self.status['root']['bt']
        key = 'footprint', la, lu, bx, bt
        self._handle('spike', 'footprint', key)

    def create(self, key, stage):
        self.line('footprint', 'info')

        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'footprint',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )

        spike = self.spike
        spike /= spike.max(axis=1, keepdims=True)

        model = self.footprint_model
        model.spike_set(spike)
        model.fit(
            lr=self.status['root']['learning-rate'],
            steps_per_epoch=self.status['root']['step'],
            epochs=self.status['root']['epoch'],
            min_delta=self.status['root']['tol'],
            batch=self.status['root']['batch'],
            log_dir=log_dir,
            stage=stage,
        )

        footprint = model.footprint.val
        footprint /= footprint.max(axis=1, keepdims=True)
        return footprint

    def save(self, base, val):
        save_numpy(base, val)
        return val
