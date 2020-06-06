import os

import tensorflow as tf
import numpy as np

from .base import Command, option
from ..footprint.clean import clean_footprint


class CleanCommand(Command):

    description = 'Clean segment'

    name = 'clean'
    options = [
        option('job-dir', flag=False, value_required=False),
        option('name', flag=False, default='default'),
        option('batch', flag=False, default=100),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        gauss = self.status['root']['gauss']
        radius = self.status['root']['diameter']
        self.key = self.status['footprint_current'] + ('clean', (gauss, radius))
        self._handle('footprint')

    def create(self):
        def gen():
            for f in self.footprint:
                yield f

        self.line('clean')
        gauss, radius = self.key[-1]
        batch = int(self.option('batch'))
        footprint, score = clean_footprint(self.footprint, self.mask, gauss, radius, batch)
        self._score = score
        return footprint

    def save(self, base, val):
        np.save(base + '.npy', val.numpy())
        np.save(base + '-score.npy', self._score.numpy())
