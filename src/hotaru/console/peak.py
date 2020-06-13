import os
from datetime import datetime

import tensorflow as tf
import pandas as pd

from .base import Command, option
from ..footprint.find import find_peak
from ..util.dataset import unmasked
from ..util.csv import save_csv


class PeakCommand(Command):

    description = 'Find peaks'

    name = 'peak'
    options = [
        option('job-dir', flag=False, value_required=False),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        gauss = self.status['root']['gauss']
        radius = self.status['root']['radius']
        thr_gl = self.status['root']['thr-gl']
        shard = self.status['root']['shard']
        key = 'peak', gauss, radius, thr_gl, shard
        self._handle(None, 'peak', key)

    def create(self, key, stage):
        self.line('peak')
        gauss, radius, thr_gl, shard = key[0][1:]
        batch = self.status['root']['batch']
        data = self.data.shard(shard, 0)
        mask = self.mask
        nt = self.status['root']['nt']
        prog = tf.keras.utils.Progbar((nt + shard - 1) // shard)
        peak = find_peak(data, mask, gauss, radius, thr_gl, batch, prog)

        ts, rs, ys, xs, gs = peak
        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'peak',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():
            tf.summary.histogram(f'radius/{stage:03d}', rs, step=0)
            tf.summary.histogram(f'laplacian/{stage:03d}', gs, step=0)
        writer.close()
        return peak

    def save(self, base, peak):
        save_csv(base, peak, ('ts', 'rs', 'ys', 'xs', 'gs'))
        return peak
