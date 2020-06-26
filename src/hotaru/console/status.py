import numpy as np


class Status(object):

    def __init__(self):
        self.params = dict()
        self.history = dict()
        self.saved = dict()

    def add_saved(self, key, name):
        self.saved[key] = name

    def find_saved(self, key):
        return self.saved.get(key, None)

    def get_params(self, stage):
        p = self.params
        if stage == 0:
            return p['imgs-file'], p['mask-type'],
        if stage == 1:
            return p['gauss'], self.radius, p['min-intensity'], p['min-distance'], p['shard'],
        if stage == 2:
            return p['thr-intensity'], p['thr-distance'],
        if stage % 3 == 0:
            return self.tau, p['lu'], p['bx'], p['bt'],
        if stage % 3 == 1:
            return p['la'], p['bx'], p['bt'],
        if stage % 3 == 2:
            return p['gauss'], self.radius, p['thr-firmness'], p['thr-sim-area'], p['thr-similarity'],

    def get_test_params(self):
        p = self.params
        return p['min-intensity'], p['max-intensity'], p['min-distance'], p['max-distance'],

    @property
    def tau(self):
        p = self.params
        return p['hz'], p['tau-rise'], p['tau-fall'], p['tau-scale'],

    @property
    def radius(self):
        radius_type = self.params['radius-type']
        r = self.params['radius']
        if radius_type == 'linear':
            out = np.linspace(r[0], r[1], int(r[2]))
        elif radius_type == 'log':
            out = np.logspace(np.log10(r[0]), np.log10(r[1]), int(r[2]))
        elif radius_type == 'manual':
            out = r
        else:
            raise RuntimeError(f'invalid radius type: {radius_type}')
        return tuple(out)
