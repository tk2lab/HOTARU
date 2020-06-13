import pickle
import os

import tensorflow as tf
import pandas as pd
import numpy as np

from cleo import Command as CommandBase
from cleo import option

from ..image.load import get_shape, load_data
from ..train.model import HotaruModel
from ..util.dataset import normalized
from ..util.npy import load_numpy
from ..util.csv import load_csv


class Command(CommandBase):

    @property
    def work_dir(self):
        return os.path.join(self.application.job_dir, 'work')

    @property
    def status(self):
        return self.application.status

    @property
    def imgs(self):

        def _gen():
            for x in imgs:
                yield tf.convert_to_tensor(wrap(x)[y0:y1, x0:x1], tf.float32)

        imgs_file = self.status['root']['imgs-file']
        imgs_file = os.path.join(self.application.job_dir, imgs_file)
        imgs, wrap = load_data(imgs_file)
        y0, y1, x0, x1 = self.status['root']['rect']
        return tf.data.Dataset.from_generator(_gen, tf.float32)

    @property
    def normalized_imgs(self):
        avgt = load_numpy(os.path.join(self.work_dir, 'avgt'))
        avgx = load_numpy(os.path.join(self.work_dir, 'avgx'))
        std = self.status['root']['std']
        return normalized(self.imgs, avgt, avgx, std)

    @property
    def data(self):
        data_file = os.path.join(self.work_dir, 'data.tfrecord')
        data = tf.data.TFRecordDataset(data_file)
        data = data.map(lambda ex: tf.io.parse_tensor(ex, tf.float32))
        return data

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
    def model(self):
        if hasattr(self.application, 'model'):
            model = self.application.model
        else:
            nk = self.clean.shape[0]
            nx = self.status['root']['nx']
            nt = self.status['root']['nt']
            model = HotaruModel(self.data, self.mask, nk, nx, nt)
            self.application.model = model
        model.set_double_exp(
            *(self.status['root'][n]
              for n in ('tau-fall', 'tau-rise', 'hz', 'tau-scale'))
        )
        model.variance.bx = self.status['root']['bx']
        model.variance.bt = self.status['root']['bt']
        model.la = self.status['root']['la']
        model.lu = self.status['root']['lu']
        model.compile()
        return model

    @property
    def footprint(self):
        return self._load('footprint', load_numpy)

    @property
    def clean(self):
        return self._load('clean', load_numpy)

    @property
    def spike(self):
        return self._load('spike', load_numpy)

    @property
    def current_key(self):
        return self.application.current_key

    @property
    def current_val(self):
        return self.application.current_val

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
