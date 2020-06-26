import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..train.footprint import FootprintModel
from ..util.numpy import load_numpy, save_numpy
from ..util.pickle import load_pickle


class FootprintCommand(Command):

    name = 'footprint'
    description = 'Update footprint'
    help = '''
'''

    options = [
        _option('job-dir'),
        option('force', 'f'),
    ]

    def is_error(self, stage):
        return stage < 4

    def is_target(self, stage):
        return stage % 3 == 1

    def force_stage(self, stage):
        return 3 * ((stage - 1) // 3) + 1

    def create(self, data, prev, curr, logs, la, bx, bt):
        tau = load_pickle(f'{prev}-tau')
        spike = load_numpy(f'{prev}-spike')
        nk = spike.shape[0]
        lu = self.status.params['lu']
        with self.application.strategy.scope():
            elems = self.get_model(data, tau, nk)
            model = FootprintModel(*elems)
            model.compile()
        model.set_penalty(la, lu, bx, bt)
        model.fit(
            spike, stage=curr[-3:],
            lr=self.status.params['learning-rate'],
            batch=self.status.params['batch'],
            steps_per_epoch=self.status.params['step'],
            epochs=self.status.params['epoch'],
            verbose=self.status.params['pbar'],
            min_delta=self.status.params['tol'],
            log_dir=logs,
        )
        save_numpy(f'{curr}-footprint', model.footprint.val)
