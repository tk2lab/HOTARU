import os

import numpy as np

from cleo import Command
from cleo import option

from ..util.pickle import load_pickle
from ..util.tfrecord import load_tfrecord


class CommandBase(Command):

    options = [
        option('tag', 't', '', False, False, False, 'default'),
        option('force', 'f', '', False, False, False, False),
    ]

    _suff = ''

    def handle(self):
        tag = self.option("tag")
        base_path = f'hotaru/{self._type}'
        os.makedirs(base_path, exist_ok=True)

        self.line(f'{self.name}: {tag}')
        base = f'{base_path}/{tag}{self._suff}'
        log_path = f'{base}_log.pickle'
        if self.option('force') or not os.path.exists(log_path):
            self._handle(base)
        else:
            self.line('...skip')

    def verbose(self):
        if self.option('quiet'):
            return 0
        v = self.option('verbose')
        if v is None:
            return 1
        elif v == 'null':
            return 2
        elif v == 'v':
            return 3
        elif v == 'vv':
            return 4
        return int(v)

    def data(self):
        tag = self.option('data-tag')
        base = f'hotaru/data/{tag}'
        log = load_pickle(f'{base}_log.pickle')
        mask = log['mask']
        nt = log['nt']
        data = load_tfrecord(f'{base}.tfrecord')
        return data, mask, nt

    def mask(self):
        tag = self.option('data-tag')
        base = f'hotaru/data/{tag}'
        log = load_pickle(f'{base}_log.pickle')
        return log['mask']

    def radius(self):
        kind = self.option('radius-kind')
        if kind == 'manual':
            return np.array(
                [float(v) for v in self.option('radius')], dtype=np.float32,
            )
        else:
            rmin = float(self.option('radius-min'))
            rmax = float(self.option('radius-max'))
            rnum = int(self.option('radius-num'))
            if kind == 'linear':
                return np.linspace(rmin, rmax, rmin, dtype=np.float32)
            elif kind == 'log':
                return np.logspace(
                    np.log10(rmin), np.log10(rmax), rnum, dtype=np.float32,
                )
        self.line(f'<error>bad radius kind</error>: {kind}')

    def used_radius(self, tag):
        base = f'hotaru/peak/{tag}'
        log = load_pickle(f'{base}_log.pickle')
        return log['radius']
