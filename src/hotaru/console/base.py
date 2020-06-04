import pickle
import os

from cleo import Command as CommandBase
from cleo import option

import tensorflow as tf
import pandas as pd
import numpy as np

from ..data.mask import load_maskfile


class Command(CommandBase):

    @property
    def work_dir(self):
        return os.path.join(self.application.job_dir, 'work')

    @property
    def status(self):
        return self.application.status

    @property
    def current_key(self):
        return self.application.current_key

    @property
    def current_val(self):
        return self.application.current_val

    @property
    def data_file(self):
        current_job_dir = self.application.job_dir
        return os.path.join(current_job_dir, 'work/data.tfrecord')

    @property
    def data(self):
        data = tf.data.TFRecordDataset(self.data_file)
        data = data.map(lambda ex: tf.io.parse_tensor(ex, tf.float32))
        return data

    @property
    def mask(self):
        if not hasattr(self.application, 'mask'):
            mask_file = os.path.join(self.work_dir, 'mask.npy')
            self.application.mask = load_maskfile(mask_file)
        return self.application.mask

    @property
    def peak(self):
        return self._load('peak', self.load_peak)

    @property
    def footprint(self):
        return self._load('footprint', self._npy_loader)

    @property
    def spike(self):
        return self._load('spike', self._npy_loader)

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
            status = dict(
                root=dict(), peak=dict(), footprint=dict(), spike=dict(),
                peak_current=None, footprint_current=None, spike_current=None,
            )
        self.application.status = status

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

    def load_peak(self, file_base):
        _peak_name = 'ts', 'rs', 'xs', 'ys', 'gs'
        _peak_type = tf.int32, tf.float32, tf.int32, tf.int32, tf.float32
        with tf.io.gfile.GFile(f'{file_base}.csv', 'r') as fp:
            peak = pd.read_csv(fp)
            peak = tuple(
                tf.convert_to_tensor(peak[k], t)
                for k, t in zip(_peak_name, _peak_type)
            )
        return peak

    def _npy_loader(self, file_base):
        with tf.io.gfile.GFile(f'{file_base}.npy', 'rb') as fp:
            val = np.load(fp)
        return val

    def _handle(self, _type):
        status = self.status[_type]
        key = self.key
        name = self.option('name')

        if key not in status:
            val = self.create()
            base = os.path.join(self.work_dir, _type, name)
            self.save(base, val)
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
