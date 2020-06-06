import os

import tensorflow as tf
import numpy as np

from .base import Command, option
from ..footprint.make import make_footprint


class MakeCommand(Command):

    description = 'Make segment'

    name = 'make'
    options = [
        option('job-dir', 'j', '', flag=False, value_required=False),
        option('name', None, '', flag=False, default='init'),
        option('batch', 'b', '', flag=False, default=100),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        self.key = self.status['peak_current']
        self._handle('footprint')

    def create(self):
        self.line('make')
        gauss = self.key[1][0]
        batch = int(self.option('batch'))
        ts, rs, ys, xs, gs = self.peak
        radius = self.status['root']['diameter']
        inv = {v: i for i, v in enumerate(radius)}
        rs = np.array([inv[r] for r in rs], np.int32)
        footprint, score = make_footprint(
            self.data, self.mask, gauss, radius, ts, rs, ys, xs, batch,
        )
        self._score = score
        return footprint

    def save(self, base, val):
        np.save(base + '.npy', val.numpy())
        np.save(base + '-score.npy', self._score.numpy())
