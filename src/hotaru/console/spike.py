import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..train.spike import SpikeModel
from ..util.numpy import load_numpy, save_numpy
from ..util.pickle import load_pickle, save_pickle


class SpikeCommand(Command):

    description = 'Update spike'

    name = 'spike'
    options = [
        _option('job-dir'),
        option('force', 'f'),
    ]

    def is_error(self, stage):
        return stage < 3

    def is_target(self, stage):
        return stage % 3 == 0

    def force_stage(self, stage):
        return 3 * ((stage - 0) // 3) + 0

    def create(self, data, prev, curr, logs, tau, lu, bx, bt):
        segment = load_numpy(f'{prev}-segment')
        nk = segment.shape[0]
        la = self.status.params['la']
        with self.application.strategy.scope():
            elems = self.get_model(data, tau, nk)
            model = SpikeModel(*elems)
            model.compile()
        model.set_penalty(la, lu, bx, bt)
        model.fit(
            segment, stage=curr[-3:],
            lr=self.status.params['learning-rate'],
            batch=self.status.params['batch'],
            steps_per_epoch=self.status.params['step'],
            epochs=self.status.params['epoch'],
            verbose=self.status.params['pbar'],
            min_delta=self.status.params['tol'],
            log_dir=logs,
        )
        save_pickle(f'{curr}-tau', tuple(tau))
        save_numpy(f'{curr}-spike', model.spike.val)
