import pickle
import os

import tensorflow.keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import GPUtil

from cleo import Command as CommandBase
from cleo import option

from ..image.load import get_shape, load_data
from ..util.dataset import normalized
from ..util.npy import load_numpy
from ..util.csv import load_csv
from ..optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from ..train.variance import Variance
from ..train.model import FootprintModel, SpikeModel


def _option(*args):
    return option(*args, flag=False, value_required=False)


class Command(CommandBase):

    @property
    def work_dir(self):
        return os.path.join(self.application.job_dir, 'work')

    @property
    def status(self):
        return self.application.status

    @property
    def data(self):
        if not hasattr(self.application, 'data'):
            data_file = os.path.join(self.work_dir, 'data.tfrecord')
            data = tf.data.TFRecordDataset(data_file)
            data = data.map(lambda ex: tf.io.parse_tensor(ex, tf.float32))
            self.application.data = data
        return self.application.data

    @property
    def mask(self):
        if not hasattr(self.application, 'mask'):
            mask_file = os.path.join(self.work_dir, 'mask')
            self.application.mask = load_numpy(mask_file)
        return self.application.mask

    @property
    def peak(self):
        name = 'ts', 'rs', 'ys', 'xs', 'gs'
        typ = np.int32, np.float32, np.int32, np.int32, np.float32
        return self._load('peak', lambda b: load_csv(b, name, typ))

    @property
    def footprint(self):
        val = self._load('footprint', load_numpy)
        self.footprint_model.footprint.val = val
        return val

    @footprint.setter
    def footprint(self, val):
        self.footprint_model.footprint.val = val

    @property
    def clean(self):
        return self._load('clean', load_numpy)

    @property
    def spike(self):
        self.ensure_model()
        val = self._load('spike', load_numpy)
        self.application._spike_model.spike.val = val
        return  val

    @spike.setter
    def spike(self, val):
        self.spike_model.spike.val = val

    @property
    def current_key(self):
        return self.application.current_key

    @property
    def current_val(self):
        return self.application.current_val

    @property
    def footprint_model(self):
        self.ensure_model()
        return self.application._footprint_model

    @property
    def spike_model(self):
        self.ensure_model()
        return self.application._spike_model

    def ensure_model(self):
        taus = tuple(
            self.status['root'][n]
            for n in ('tau-fall', 'tau-rise', 'hz', 'tau-scale')
        )
        if not hasattr(self.application, '_spike_model'):
            nk = self.status['root']['nk']
            nx = self.status['root']['nx']
            nt = self.status['root']['nt']
            with self.application.strategy.scope():
                variance = Variance(self.data, nk, nx, nt)
                variance.set_double_exp(*taus)
                footprint = Input(nk, nx, name='footprint')
                footprint.mask = self.mask
                spike = Input(nk, variance.nu, name='spike')
                footprint_model = FootprintModel(footprint, spike, variance)
                spike_model = SpikeModel(footprint, spike, variance)
                footprint_model.compile()
                spike_model.compile()
            self.application._footprint_model = footprint_model
            self.application._spike_model = spike_model
        else:
            self.application._spike_model.variance.set_double_exp(*taus)
        variance = self.application._footprint_model.variance
        bx = self.status['root']['bx']
        bt = self.status['root']['bt']
        variance.set_baseline(bx, bt)
        la = self.status['root']['la']
        lu = self.status['root']['lu']
        nm = K.get_value(variance._nm)
        self.application._footprint_model.footprint.l = lu / nm
        self.application._spike_model.spike.l = lu /nm

    def set_job_dir(self, default='.'):
        current_job_dir = self.application.job_dir
        job_dir = self.option('job-dir')
        if current_job_dir is None:
            self.application.job_dir = job_dir or default
            self.load_status()
        elif job_dir and job_dir != current_job_dir:
            raise RuntimeError('config mismatch: job-dir')

    def load_status(self):
        status_file = os.path.join(self.work_dir, 'status.pickle')
        if tf.io.gfile.exists(status_file):
            with tf.io.gfile.GFile(status_file, 'rb') as fp:
                status = pickle.load(fp)
        else:
            tf.io.gfile.makedirs(os.path.join(self.work_dir, 'peak'))
            tf.io.gfile.makedirs(os.path.join(self.work_dir, 'footprint'))
            tf.io.gfile.makedirs(os.path.join(self.work_dir, 'spike'))
            tf.io.gfile.makedirs(os.path.join(self.work_dir, 'clean'))
            status = dict(
                root=dict(), peak=dict(), footprint=dict(),
                spike=dict(), clean=dict(),
                peak_current=None, footprint_current=None,
                spike_current=None, clean_current=None,
            )
        self.application.status = status

    def print_gpu_memory(self):
        for g in GPUtil.getGPUs():
            self.line(f'{g.memoryUsed}')

    def save_status(self):
        status_file = os.path.join(self.work_dir, 'status.pickle')
        with tf.io.gfile.GFile(status_file, 'w') as fp:
            pickle.dump(self.status, fp)

    def _load(self, _type, loader):
        if self.current_key[_type] is None:
            key = self.status[_type + '_current']
            name = self.status[_type][key]
            file_base = os.path.join(self.work_dir, _type, name)
            val = loader(file_base)
            self.current_key[_type] = key
            self.current_val[_type] = val
        return self.current_val[_type]

    def _handle(self, prev, _type, key):
        key = (key,)
        if prev is not None:
            if self.option('prev'):
                self.status[f'{prev}_current'] = {
                    v: k for k, v in self.status[prev].items()
                }[self.option('prev')]
            key = self.status[f'{prev}_current'] + key
        status = self.status[_type]
        name = self.status['root']['name']

        if self.option('force') or key not in status:
            stage = len(key)
            base = os.path.join(self.work_dir, _type, name)
            val = self.create(key, stage)
            val = self.save(base, val)
            dup_keys = tuple(k for k, v in status.items() if v == name)
            for k in dup_keys:
                del status[k]
            status[key] = name
            self.current_key[_type] = key
            self.current_val[_type] = val

        if self.status[f'{_type}_current'] != key:
            self.status[f'{_type}_current'] = key
            self.save_status()

        if key != self.current_key[_type]:
            self.current_key[_type] = None
            self.current_val[_type] = None
