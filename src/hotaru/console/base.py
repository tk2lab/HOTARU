import os

from cleo import Command
from cleo import option

from ..util.pickle import load_pickle
from ..util.tfrecord import load_tfrecord


class CommandBase(Command):

    options = [
        option('tag', 't', '', False, False, False, 'default'),
        option('force', 'f', '', False, False, False, False),
    ]

    optimizer_options = [
        option('lr', 'l', '', False, False, False, 0.01),
        option('tol', 'o', '', False, False, False, 0.001),
        option('epoch', None, '', False, False, False, 100),
        option('step', None, '', False, False, False, 100),
        option('batch', None, '', False, False, False, 100),
    ]

    def handle(self):
        tag = self.option("tag")
        base_path = f'hotaru/{self._type}'
        os.makedirs(base_path, exist_ok=True)

        base = f'{base_path}/{tag}'
        log_path = f'{base}_log.pickle'
        if self.option('force') or not os.path.exists(log_path):
            self._handle(base)

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

    def used_radius(self):
        tag = self.option('peak-tag')
        base = f'hotaru/peak/{tag}'
        log = load_pickle(f'{base}_log.pickle')
        return log['radius']
