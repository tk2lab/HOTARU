import os

import tensorflow as tf
import numpy as np

from .base import Command, option
from ..data.dataset import unmasked
from ..footprint.make import make_footprint


class MakeCommand(Command):

    description = 'Make segment'

    name = 'make'
    options = [
        option('job-dir', flag=False, value_required=False),
        option('name', flag=False, default='init'),
        option('batch', flag=False, default=100),
    ]

    def handle(self):
        self.set_job_dir()
        self.key = self.status['peak_current']
        self._handle('footprint')

    def create(self):
        self.line('make')
        data = unmasked(self.data, self.mask)
        gauss = self.key[1][0]
        batch = int(self.option('batch'))
        ts, rs, ys, xs, gs = self.peak
        footprint, score = make_footprint(
            data, self.mask, gauss, ts, rs, ys, xs, batch,
        )
        self._score = score
        return footprint

    def save(self, base, val):
        np.save(base + '.npy', val.numpy())
        np.save(base + '-score.npy', self._score.numpy())
