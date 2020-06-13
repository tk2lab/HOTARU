import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base import Command, option
from ..footprint.reduce import reduce_peak
from ..footprint.make import make_footprint
from ..train.callback import summary_footprint, summary_footprint_stat
from ..util.npy import save_numpy
from ..util.csv import save_csv


class SegmentCommand(Command):

    description = 'Make segment'

    name = 'segment'
    options = [
        option('job-dir', 'j', '', flag=False, value_required=False),
        option('prev', flag=False, value_required=False),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        thr_dist = self.status['root']['thr-dist']
        key = 'segment', thr_dist
        self._handle('peak', 'clean', key)

    def create(self, key, stage):
        self.line('segment')
        gauss, radius, thr_gl, shard = key[-2][1:]
        thr_dist = key[-1][1]
        batch = self.status['root']['batch']
        peak = reduce_peak(self.peak, thr_dist)
        ts, rs, ys, xs, gs = peak
        inv = {v: i for i, v in enumerate(radius)}
        rs_id = np.array([inv[r] for r in rs], np.int32)
        peak_id = ts, rs_id, ys, xs, gs
        data = self.data.shard(shard, 0)
        mask = self.mask
        footprint = make_footprint(data, mask, gauss, radius, peak_id, batch)

        mask = self.mask
        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'segment',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():
            tf.summary.histogram(f'size/{stage:03d}', footprint.sum(axis=1), step=0)
            tf.summary.histogram(f'radius/{stage:03d}', rs, step=0)
            tf.summary.histogram(f'laplacian/{stage:03d}', gs, step=0)
            summary_footprint_stat(footprint, mask, stage)
            summary_footprint(footprint, mask, stage)
        writer.close()

        return footprint, peak

    def save(self, base, val):
        footprint, peak = val
        save_numpy(base, footprint)
        save_csv(base, peak, ('ts', 'rs', 'ys', 'xs', 'gs'))
        return footprint
