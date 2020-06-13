import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base import Command, option
from ..footprint.clean import clean_footprint
from ..train.callback import summary_footprint, summary_footprint_stat
from ..util.npy import save_numpy
from ..util.csv import save_csv


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
        radius = self.status['root']['radius']
        key = self.status['footprint_current'] + (('clean', gauss, radius),)
        self._handle('clean', key)

    def create(self, key, stage):
        def gen():
            for f in self.footprint:
                yield f

        self.line('clean')
        mask = self.mask
        gauss, radius = key[-1][1:]
        batch = int(self.option('batch'))
        footprint, rs, ys = clean_footprint(
            self.footprint, mask, gauss, radius, batch,
        )
        rs, fs, ys, xs = rs[:, 0], rs[:, 1], ys[:, 0], ys[:, 1]

        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'clean',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():
            tf.summary.histogram(f'radius{stage}', rs, step=0)
            tf.summary.histogram(f'firmness{stage}', fs, step=0)
            summary_footprint_stat(footprint, mask, stage)
            summary_footprint(footprint, mask, stage)
        writer.close()

        footprint *= fs[:, None]
        return footprint, (rs, ys, xs, fs)

    def save(self, base, val):
        footprint, peak = val
        save_numpy(base, footprint)
        save_csv(base, peak, ('rs', 'ys', 'xs', 'fs'))
        return footprint
