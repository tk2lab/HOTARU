import tensorflow as tf
import pandas as pd
import numpy as np

from .base import Command, option, _option
from ..footprint.find import find_peak
from ..util.dataset import unmasked
from ..util.tfrecord import load_tfrecord
from ..util.numpy import load_numpy, save_numpy
from ..util.pickle import load_pickle, save_pickle


class PeakCommand(Command):

    description = 'Find peaks'

    name = 'peak'
    options = [
        _option('job-dir'),
        option('force', 'f'),
    ]

    def is_error(self, stage):
        return stage < 1

    def is_target(self, stage):
        return stage == 1

    def force_stage(self, stage):
        return 1

    def create(self, data, prev, curr, logs, gauss, radius, thr_gl, shard):
        tfrecord = load_tfrecord(f'{data}-data')
        mask = load_numpy(f'{data}-mask')
        nt = load_pickle(f'{data}-stat')[1]
        batch = self.status.params['batch']
        pos, score = find_peak(
            tfrecord, mask, gauss, radius, thr_gl, shard, batch, nt,
        )
        save_pickle(f'{curr}-filter', (gauss, radius, shard))
        save_numpy(f'{curr}-peak', pos)
        save_numpy(f'{curr}-intensity', score)
