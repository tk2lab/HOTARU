import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base import Command, option
from ..footprint.clean import clean_footprint
from ..train.callback import summary_footprint, summary_footprint_stat, normalized_and_sort
from ..util.npy import save_numpy
from ..util.csv import save_csv


class CleanCommand(Command):

    description = 'Clean segment'

    name = 'clean'
    options = [
        option('job-dir', flag=False, value_required=False),
        option('prev', flag=False, value_required=False),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        gauss = self.status['root']['gauss']
        rmin = self.status['root']['radius-min']
        rmax = self.status['root']['radius-max']
        rnum = self.status['root']['radius-num']
        key = 'clean', gauss, rmin, rmax, rnum
        self._handle('footprint', 'clean', key)

    def create(self, key, stage):
        def gen():
            for f in self.footprint:
                yield f

        self.line('clean')
        mask = self.mask
        gauss, rmin, rmax, rnum = key[-1][1:]
        radius = tuple(np.linspace(rmin, rmax, rnum))
        batch = self.status['root']['batch']
        footprint, rs, ys = clean_footprint(
            self.footprint, mask, gauss, radius, batch,
        )
        rs, fs, ys, xs = rs[:, 0], rs[:, 1], ys[:, 0], ys[:, 1]
        idx = np.argsort(fs)[::-1]
        rs = rs[idx]
        fs = fs[idx]
        ys = ys[idx]
        xs = xs[idx]
        footprint = footprint[idx]

        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'clean',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():
            fsum = footprint.sum(axis=1)
            tf.summary.histogram(f'size/{stage:03d}', fsum, step=0)
            tf.summary.histogram(f'radius/{stage:03d}', rs, step=0)
            tf.summary.histogram(f'firmness/{stage:03d}', fs, step=0)
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
