import os

import tensorflow as tf
import numpy as np

from cleo import Command as CommandBase
from cleo import option

from .status import Status
from ..util.dataset import normalized
from ..util.tfrecord import load_tfrecord
from ..util.numpy import load_numpy
from ..util.pickle import load_pickle, save_pickle

from ..optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from ..train.variance import Variance


def _option(*args):
    return option(*args, flag=False, value_required=False)


class Command(CommandBase):

    def set_job_dir(self, default='.'):
        current_job_dir = self.application.job_dir
        job_dir = self.option('job-dir')
        if current_job_dir is None:
            self.application.job_dir = job_dir or default
            status_base = os.path.join(self.work_dir, 'status')
            if tf.io.gfile.exists(f'{status_base}.pickle'):
                self.application.status = load_pickle(status_base)
            else:
                tf.io.gfile.makedirs(self.work_dir)
                self.application.status = Status()
        elif job_dir and job_dir != current_job_dir:
            raise RuntimeError('config mismatch: job-dir')

    @property
    def work_dir(self):
        return os.path.join(self.application.job_dir, 'work')

    @property
    def logs_dir(self):
        return os.path.join(self.application.job_dir, 'logs')

    @property
    def status(self):
        return self.application.status

    def save_status(self):
        status_base = os.path.join(self.work_dir, 'status')
        save_pickle(status_base, self.status)

    def get_model(self, data, tau, nk):
        tfrecord = load_tfrecord(f'{data}-data')
        mask = load_numpy(f'{data}-mask')
        nx, nt = load_pickle(f'{data}-stat')[:2]
        variance = Variance(tfrecord, nk, nx, nt)
        variance.set_double_exp(*tau)
        footprint = Input(nk, nx, name='footprint')
        footprint.mask = mask
        spike = Input(nk, variance.nu, name='spike')
        return footprint, spike, variance

    def handle(self):
        self.set_job_dir()
        name = self.status.params['name']
        history = self.status.history.get(name, ())
        stage = len(history)
        if self.option('force'):
            history = history[:self.force_stage(stage)]
            stage = len(history)
        if self.is_error(stage):
            self.line(f'invalid stage: {self.name} {stage}', 'error')
            return 1
        elif self.is_target(stage):
            self.line(f'{self.name} ({stage})', 'info')
            key = self.status.get_params(stage)
            history = history + (key,)
            curr = self.status.find_saved(history)
            if self.option('force') or not curr:
                data = self.status.find_saved(history[:1])
                if data:
                    data = os.path.join(self.work_dir, data, '000')
                prev = self.status.find_saved(history[:-1])
                if prev:
                    prev = os.path.join(self.work_dir, prev, f'{stage-1:03d}')
                curr = os.path.join(self.work_dir, name, f'{stage:03d}')
                logs = os.path.join(self.logs_dir, name, f'{stage:03d}')
                tf.io.gfile.makedirs(os.path.join(self.work_dir, name))
                self.create(data, prev, curr, logs, *key)
                self.status.add_saved(history, name)
            self.status.history[name] = history
            self.save_status()
            return 0
