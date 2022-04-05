import os

import numpy as np
import click

from hotaru.util.tfrecord import load_tfrecord
from hotaru.util.pickle import load_pickle
from hotaru.util.pickle import save_pickle
from hotaru.util.numpy import load_numpy
from hotaru.util.numpy import save_numpy
from hotaru.util.csv import load_csv
from hotaru.util.csv import save_csv
from hotaru.util.tiff import save_tiff


class Obj(dict):

    def __init__(self):
        self._log = {}

    def __getattr__(self, key):
        return self.get(key)

    def need_exec(self): 
        kind = self.kind
        if self.force:
            return True
        if kind == 'output':
            return True
        if kind in ('temporal', 'spatial', 'clean'):
            if not isinstance(self.stage, int):
                return True
        path = self.log_path()
        return not os.path.exists(path)

    @property
    def data(self):
        path = self.out_path('data', self.data_tag, '')
        return load_tfrecord(f'{path}.tfrecord')

    @property
    def peak(self):
        path = self.out_path('peak', self.find_tag, '_find')
        return load_csv(f'{path}.csv')

    @property
    def peak_trial(self):
        path = self.out_path('peak', self.tag, '')
        return load_csv(f'{path}.csv')

    @property
    def segment(self):
        path = self.out_path('segment', self.segment_tag, self.segment_stage)
        if self.segment_stage == '_curr':
            if not os.path.exists(f'{path}.npy'):
                path = self.out_path('segment', self.init_tag, '_000')
        return load_numpy(f'{path}.npy')

    @property
    def index(self):
        path = self.out_path('peak', self.segment_tag, self.segment_stage)
        if self.segment_stage == '_curr':
            if not os.path.exists(f'{path}.csv'):
                path = self.out_path('peak', self.init_tag, '_000')
        return load_csv(f'{path}.csv').query('accept == "yes"').index

    @property
    def spike(self):
        path = self.out_path('spike', self.spike_tag, self.spike_stage)
        return load_numpy(f'{path}.npy')

    @property
    def footprint(self):
        path = self.out_path('footprint', self.footprint_tag, self.footprint_stage)
        return load_numpy(f'{path}.npy')

    def num_cell(self, stage):
        if stage == -2:
            path = self.out_path('segment', self.init_tag, 0)
        elif stage == -1:
            path = self.out_path('segment', self.tag, -1)
        else:
            path = self.out_path('segment', self.tag, stage)
        return load_numpy(f'{path}.npy').shape[0]

    @property
    def hz(self):
        return self.log('data', self.data_tag, '')['hz']

    @property
    def mask(self):
        return self.log('data', self.data_tag, '')['mask']

    @property
    def avgx(self):
        return self.log('data', self.data_tag, '')['avgx']

    @property
    def nx(self):
        return self.log('data', self.data_tag, '')['mask'].sum()

    @property
    def nt(self):
        return self.log('data', self.data_tag, '')['nt']

    @property
    def used_radius_min(self):
        return self.log('find', self.find_tag, '')['radius_min']

    @property
    def used_radius_max(self):
        return self.log('find', self.find_tag, '')['radius_max']

    @property
    def used_distance(self):
        return self.log('test', self.tag, '')['distance']

    @property
    def used_tau(self):
        log = self.log('temporal', self.spike_tag, self.spike_stage)
        return dict(
            hz=self.hz,
            tau1=log['tau_rise'],
            tau2=log['tau_fall'],
            tscale=log['tau_scale'],
        )

    @property
    def radius(self):
        if self.radius_type == 'linear':
            return np.linspace(self.radius_min, self.radius_max, self.radius_num)
        elif self.radius_type == 'log':
            return np.logspace(np.log10(self.radius_min), np.log10(self.radius_max), self.radius_num)

    @property
    def tau(self):
        return dict(
            hz=self.hz,
            tau1=self.tau_rise,
            tau2=self.tau_fall,
            tscale=self.tau_scale,
        )

    @property
    def reg(self):
        return dict(
            la=self.la,
            lu=self.lu,
            bx=self.bx,
            bt=self.bt,
        )

    @property
    def opt(self):
        return dict(
            lr=self.lr,
            min_delta=self.tol,
            epochs=self.epoch,
            steps_per_epoch=self.steps,
            batch=self.batch,
        )

    def out_path(self, kind, tag=None, stage=None):
        if tag is None:
            tag = self.tag
        if stage is None:
            stage = self.stage
        if stage is None:
            stage = ''
        elif isinstance(stage, int):
            if stage == -1:
                stage = '_curr'
            else:
                stage = f'_{stage:03}'
        os.makedirs(f'{self.workdir}/{kind}', exist_ok=True)
        return f'{self.workdir}/{kind}/{tag}{stage}'

    def log(self, kind, tag, stage):
        key = kind, tag, stage
        if key not in self._log:
            if stage is None:
                stage = ''
            elif isinstance(stage, int):
                stage = f'_{stage:03}'
            path = f'{self.workdir}/log/{tag}{stage}_{kind}.pickle'
            self._log[key] = load_pickle(path)
        return self._log[key]

    def log_path(self):
        kind = self.kind
        if kind in ('temporal', 'spatial', 'clean', 'output'):
            tag = self.tag
            stage = self.stage
        else:
            tag = self.tag
            stage = None
            if kind == 'data':
                if self.data_tag is not None:
                    tag = self.data_tag
            if kind == 'find':
                if self.find_tag is not None:
                    tag = self.find_tag
            if kind == 'init':
                if self.init_tag is not None:
                    tag = self.init_tag
        if stage is None:
            stage = ''
        elif isinstance(stage, int):
            stage = f'_{stage:03}'
        os.makedirs(f'{self.workdir}/log', exist_ok=True)
        return f'{self.workdir}/log/{tag}{stage}_{kind}.pickle'

    def save_numpy(self, data, kind, tag=None, stage=None):
        out_path = self.out_path(kind, tag, stage)
        save_numpy(f'{out_path}.npy', data)

    def save_csv(self, data, kind, tag=None, stage=None):
        out_path = self.out_path(kind, tag, stage)
        save_csv(f'{out_path}.csv', data)

    def save_tiff(self, data, kind, tag=None, stage=None):
        out_path = self.out_path(kind, tag, stage)
        save_tiff(f'{out_path}.tif', data)

    def save_log(self, log):
        path = self.log_path()
        save_pickle(path, log)
