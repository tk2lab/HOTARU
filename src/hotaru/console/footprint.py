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
        self.line('footprint')
        self.print_gpu_memory()
        model = self.model
        model.spike.val = self.spike
        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'footprint',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
        self.print_gpu_memory()
        model.update_footprint(
            lr=self.status['root']['learning-rate'],
            steps_per_epoch=self.status['root']['step'],
            epochs=self.status['root']['epoch'],
            min_delta=self.status['root']['tol'],
            batch=self.status['root']['batch'],
            log_dir=log_dir,
            stage=stage,
        )
        self.print_gpu_memory()
        return model.footprint.val

    def save(self, base, val):
        save_numpy(base, val)
        return val
