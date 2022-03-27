import os

import numpy as np

from cleo import Command
from cleo import option

from ..util.timer import Timer

from .options import option_type
from ..util.tfrecord import load_tfrecord
from ..util.pickle import load_pickle
from ..util.pickle import save_pickle


class CommandBase(Command):

    @staticmethod
    def base_options(default=''):
        return [
        option('tag', 't', '', False, False, False, 'default'),
        option('suffix', 's', '', False, False, False, default),
        option('force', 'f', '', False, False, False, False),
    ]

    _suff = ''

    def p(self, name):
        return option_type.get(name, str)(self.option(name))

    def handle(self):
        base_path = f'hotaru/{self._type}'
        os.makedirs(base_path, exist_ok=True)

        tag = self.option('tag') + self.option('suffix')
        self.line(f'{self.name}: {tag}')
        base = f'{base_path}/{tag}{self._suff}'
        log_path = f'{base}_log.pickle'
        if self.option('force') or not os.path.exists(log_path):
            options = {
                o.long_name: self.p(o.long_name)
                for o in self.options
            }
            options['verbose'] = self.verbose()
            with Timer() as timer:
                self._handle(base, options)
            options['time'] = timer.get()
            save_pickle(log_path, options)
            self.call(f'fig{self.name}', f'--tag {tag}')
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

    def radius(self):
        kind = self.p('radius-kind')
        if kind == 'manual':
            return np.array(self.p('radius-elem'), np.float32)
        else:
            rmin = self.p('radius-min')
            rmax = self.p('radius-max')
            rnum = self.p('radius-num')
            if kind == 'linear':
                return np.linspace(rmin, rmax, rmin, dtype=np.float32)
            elif kind == 'log':
                return np.logspace(
                    np.log10(rmin), np.log10(rmax), rnum, dtype=np.float32,
                )
        self.line_error(f'bad radius kind: {kind}')

    def data(self, tag=None):
        if tag is None:
            tag = self.option('data-tag')
        base = f'hotaru/data/{tag}'
        data = load_tfrecord(f'{base}.tfrecord')
        return data

    def data_prop(self, tag=None, avgx=False):
        if tag is None:
            tag = self.option('data-tag')
        base = f'hotaru/data/{tag}'
        log = load_pickle(f'{base}_log.pickle')
        if avgx:
            return log['mask'], log['nt'], log['avgx'] / log['sstd']
        else:
            return log['mask'], log['nt']
