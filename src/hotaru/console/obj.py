import os

import click

from hotaru.util.tfrecord import load_tfrecord
from hotaru.util.pickle import load_pickle
from hotaru.util.pickle import save_pickle
from hotaru.util.numpy import load_numpy
from hotaru.util.numpy import save_numpy
from hotaru.util.csv import load_csv
from hotaru.util.csv import save_csv


class Obj:

    def __init__(self):
        self._log = {}

    def data(self):
        path = self.out_path('data', self.data_tag, '')
        return load_tfrecord(f'{path}.tfrecord')

    def peak(self, initial=False):
        if initial:
            tag = self.find_tag
            stage = '_find'
        else:
            tag = self.prev_tag
            stage = self.prev_stage
        path = self.out_path('peak', tag, stage)
        return load_csv(f'{path}.csv')

    def segment(self, initial=False):
        if initial:
            tag = self.init_tag
            stage = '_init'
        else:
            tag = self.prev_tag
            stage = self.prev_stage
        path = self.out_path('segment', tag, stage)
        print(path)
        return load_numpy(f'{path}.npy')

    def index(self, initial=False):
        if initial:
            tag = self.init_tag
            stage = '_init'
        else:
            tag = self.prev_tag
            stage = self.prev_stage
        path = self.out_path('peak', tag, stage)
        return load_csv(f'{path}.csv').query('accept == "yes"').index

    def spike(self):
        path = self.out_path('spike', self.prev_tag, self.prev_stage)
        return load_numpy(f'{path}.npy')

    def footprint(self):
        path = self.out_path('footprint', self.prev_tag, self.prev_stage)
        return load_numpy(f'{path}.npy')

    def log(self, kind, tag=None, stage=None):
        if tag is None:
            if kind == 'data':
                tag = self.data_tag
            else:
                tag = self.prev_tag
        if stage is None:
            stage = self.prev_stage
        key = kind, tag, stage
        if key not in self._log:
            path = self.log_path(*key)
            self._log[key] = load_pickle(path)
        return self._log[key]

    def mask(self):
        return self.log('data', self.data_tag, '')['mask']

    def avgx(self):
        return self.log('data', self.data_tag, '')['avgx']

    def nx(self):
        return self.log('data', self.data_tag, '')['mask'].sum()

    def nt(self):
        return self.log('data', self.data_tag, '')['nt']

    def radius(self):
        return self.log('find', self.prev_tag, '')['radius']

    def radius_min(self):
        return self.log('find', self.prev_tag, '')['radius'][0]

    def radius_max(self):
        return self.log('find', self.prev_tag, '')['radius'][-1]

    def out_path(self, kind, tag=None, stage=None):
        if tag is None:
            tag = self.tag
        if stage is None:
            stage = self.stage
        if stage is None:
            stage = ''
        elif isinstance(stage, int):
            stage = f'_{stage:03}'
        os.makedirs(f'{self.workdir}/{kind}', exist_ok=True)
        return f'{self.workdir}/{kind}/{tag}{stage}'

    def log_path(self, kind, tag=None, stage=None):
        if tag is None:
            tag = self.tag
        if stage is None:
            stage = self.stage
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

    def need_exec(self, kind): 
        if self.force:
            return True
        if (kind in ('temporal', 'spatiol', 'clean')) and (self.stage == ''):
            return True
        elif kind == 'data' and self.data_tag is not None:
            tag = self.data_tag
            stage = ''
        elif kind == 'find' and self.find_tag is not None:
            tag = self.find_tag
            stage = ''
        elif kind == 'init' and self.init_tag is not None:
            tag = self.init_tag
            stage = ''
        else:
            tag = self.tag
            stage = self.stage
        return not os.path.exists(self.log_path(kind, tag, stage))

    def save_log(self, kind, log):
        path = self.log_path(kind)
        save_pickle(path, log)
