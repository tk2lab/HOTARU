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
        option('name', flag=False, default='default'),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        gauss = self.status['root']['gauss']
        radius = self.status['root']['radius']
        thr_gl = self.status['root']['thr-gl']
        key = (('peak', gauss, radius, thr_gl),)
        self._handle('peak', key)

    def create(self, key, stage):
        self.line('peak')
        gauss, radius, thr_gl = key[0][1:]
        batch = self.status['root']['batch']
        nt = self.status['root']['nt']
        prog = tf.keras.utils.Progbar(nt)
        peak = find_peak(
            self.data, self.mask, gauss, radius, thr_gl, batch, prog,
        )

        ts, rs, ys, xs, gs = peak
        log_dir = os.path.join(
            self.application.job_dir, 'logs', 'peak',
            datetime.now().strftime('%Y%m%d-%H%M%S'),
        )
        writer = tf.summary.create_file_writer(log_dir)
        with writer.as_default():
            tf.summary.histogram(f'radius{stage}', rs, step=0)
            tf.summary.histogram(f'laplacian{stage}', gs, step=0)
        writer.close()
        return peak

    def save(self, base, peak):
        save_csv(base, peak, ('ts', 'rs', 'ys', 'xs', 'gs'))
        return peak
