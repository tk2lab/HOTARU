import os
from datetime import datetime

import tensorflow as tf
import pandas as pd
import numpy as np

from .base import Command, option, _option
from ..footprint.find import find_peak
from ..footprint.reduce import reduce_peak
from ..util.dataset import unmasked
from ..util.csv import save_csv


class PeakCommand(Command):

    description = 'Find peaks'

    name = 'peak'
    options = [
        _option('job-dir'),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        gauss = self.status['root']['gauss']
        rmin = self.status['root']['radius-min']
        rmax = self.status['root']['radius-max']
        rnum = self.status['root']['radius-num']
        thr_gl = self.status['root']['thr-gl']
        shard = self.status['root']['shard']
        key = 'peak', gauss, rmin, rmax, rnum, thr_gl, shard
        self._handle(None, 'peak', key)

    def create(self, key, stage):
        self.line(f'peak ({stage})', 'info')

        gauss, rmin, rmax, rnum, thr_gl, shard = key[0][1:]
        data = self.data.shard(shard, 0)
        mask = self.mask
        radius = tuple(
            0.001 * round(1000 * x) for x in np.linspace(rmin, rmax, rnum)
        )
        batch = self.status['root']['batch']
        nt = self.status['root']['nt']
        prog = tf.keras.utils.Progbar((nt + shard - 1) // shard)

        peak = find_peak(
            data, mask, gauss, radius, thr_gl, batch, prog,
        )
        return peak

    def save(self, base, peak):
        save_csv(base, peak, ('ts', 'rs', 'ys', 'xs', 'gs'))
        return peak
