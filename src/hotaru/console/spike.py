import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..util.npy import save_numpy


class SpikeCommand(Command):

    description = 'Update spike'

    name = 'spike'
    options = [
        _option('job-dir'),
        _option('prev'),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        tau1 = self.status['root']['tau-fall']
        tau2 = self.status['root']['tau-rise']
        hz = self.status['root']['hz']
        tauscale = self.status['root']['tau-scale']
        bx = self.status['root']['bx']
        bt = self.status['root']['bt']
        la = self.status['root']['la']
        lu = self.status['root']['lu']
        key = 'spike', tau1, tau2, hz, tauscale, la, lu, bx, bt
        self._handle('clean', 'spike', key)

    def create(self, key, stage):
        self.line('spike', 'info')

        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'spike',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )

        model = self.spike_model
        model.footprint_set(self.clean)
        model.fit(
            lr=self.status['root']['learning-rate'],
            steps_per_epoch=self.status['root']['step'],
            epochs=self.status['root']['epoch'],
            min_delta=self.status['root']['tol'],
            batch=self.status['root']['batch'],
            log_dir=log_dir,
            stage=stage,
        )
        return model.spike.val

    def save(self, base, val):
        save_numpy(base, val)
        return val
