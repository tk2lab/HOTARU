import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..footprint.clean import clean_footprint
from ..train.summary import normalized_and_sort
from ..train.summary import summary_footprint
from ..util.npy import save_numpy
from ..util.csv import save_csv


class CleanCommand(Command):

    description = 'Clean segment'

    name = 'clean'
    options = [
        _option('job-dir'),
        _option('prev'),
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
        self.line('<info>clean</info>')

        # clean
        mask = self.mask
        gauss, rmin, rmax, rnum = key[-1][1:]
        radius = tuple(np.linspace(rmin, rmax, rnum))
        batch = self.status['root']['batch']
        footprint, rs, ys = clean_footprint(
            self.footprint, mask, gauss, radius, batch,
        )
        rs, fs, ys, xs = rs[:, 0], rs[:, 1], ys[:, 0], ys[:, 1]

        # sort
        idx = np.argsort(fs)[::-1]
        rs = rs[idx]
        fs = fs[idx]
        ys = ys[idx]
        xs = xs[idx]
        footprint = footprint[idx]

        # log
        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'clean',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():
            tf.summary.histogram(f'radius/{stage:03d}', rs, step=0)
            tf.summary.histogram(f'firmness/{stage:03d}', fs, step=0)

            fsum = footprint.sum(axis=1)
            tf.summary.histogram(f'size/{stage:03d}', fsum, step=0)
            #summary_footprint_stat(footprint, mask, stage)
            summary_footprint(footprint, mask, stage)

            # remove
            cond = fs > self.status['root']['thr-firmness']
            cond &= (radius[0] < rs) & (rs < radius[-1])
            rs, ys, xs, fs = map(lambda s: s[cond], (rs, ys, xs, fs))
            tf.summary.histogram(f'radius/{stage:03d}', rs, step=1)
            tf.summary.histogram(f'firmness/{stage:03d}', fs, step=1)

            footprint = footprint[cond]
        writer.close()

        return footprint, (rs, ys, xs, fs)

    def save(self, base, val):
        footprint, peak = val
        save_numpy(base, footprint)
        save_csv(base, peak, ('rs', 'ys', 'xs', 'fs'))
        return footprint
