from datetime import datetime
import os

import tensorflow as tf
import numpy as np

from .base import Command, option


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
        la = self.status['root']['la']
        lu = self.status['root']['lu']
        bx = self.status['root']['bx']
        bt = self.status['root']['bt']
        prev_key = self.status['spike_current']
        if self.option('prev'):
            prev_key = {v: k for k, v in self.status['spike'].items()}[self.option('prev')]
        self.key = prev_key + ('footprint', (la, lu, bx, bt))
        self._handle('footprint')

    def create(self):
        print('footprint')
        model = self.model
        model.spike.val = self.spike
        log_dir = os.path.join(self.application.job_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        model.update_footprint(
            batch=int(self.option('batch')),
            callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir, update_freq='batch'),
            ],
        )
        return model.footprint.val

    def save(self, base, val):
        np.save(base + '.npy', val)
