import numpy as np
import pandas as pd

from .base import CommandBase
from .radius import RadiusMixin
from .base import option

from ..footprint.clean import clean_footprint
from ..util.numpy import load_numpy
from ..util.numpy import save_numpy
from ..util.pickle import save_pickle


class CleanCommand(CommandBase, RadiusMixin):

    name = 'clean'
    _type = 'footprint'
    description = 'Clean segment'
    help = '''The clean command 
'''

    options = CommandBase.options + [
        option('data-tag', 'd', '', False, False, False, 'default'),
        option('footprint-tag', 'p', '', False, False, False, 'default'),
    ] + RadiusMixin._options + [
        option('thr-area-abs', None, '', False, False, False, 100),
        option('thr-area-rel', None, '', False, False, False, 2.0),
        option('batch', 'b', '', False, False, False, 100),
    ]

    def _handle(self, base):
        footprint_tag = self.option('footprint-tag')
        footprint_base = f'hotaru/footprint/{footprint_tag}'
        footprint = load_numpy(f'{footprint_base}.npy')

        mask = self.mask()
        radius = self.radius()
        batch = self.option('batch')
        verbose = self.verbose()

        footprint, peaks = clean_footprint(
            footprint, mask, radius, batch, verbose,
        )

        idx = np.argsort(peaks['firmness'].values)[::-1]
        footprint = footprint[idx]
        peaks = peaks.iloc[idx].copy()
        area = np.sum(footprint > 0.5, axis=1)
        peaks['area'] = area
        peaks.reset_index()

        thr_abs = self.option('thr-area-abs')
        thr_rel = self.option('thr-area-rel')
        x = peaks['radius']
        cond = (radius[0] < x) & (x < radius[-1])
        cond &= (area <= thr_abs + thr_rel * np.pi * x ** 2)
        peaks['accepted'] = False
        peaks.loc[cond, 'accepted'] = True
        old_nk = footprint.shape[0]
        nk = np.count_nonzero(cond)
        self.line(f'ncell: {old_nk} -> {nk}', 'comment')
        peaks.to_csv(f'{base}_peaks.csv')
        save_numpy(f'{base}.npy', footprint[cond])
        save_numpy(f'{base}_removed.npy', footprint[~cond])

        if np.any(cond):
            for l in str(peaks[cond]).split('\n'):
                self.line(l)
        if np.any(~cond):
            for l in str(peaks[~cond]).split('\n'):
                self.line(l)
        '''
        self.line(f'id, pos, firmness, rad, size')
        for i in peaks.index: 
            x, y, g, r, a = peaks.loc[i, ['x', 'y', 'firmness', 'radius', 'area']]
            self.line(f'{i:5}, ({x:>4},{y:>4}), {g:.3f}, {r:7.3f}, {a:5}')
            if not cond[i]:
                self.line('-------')
        '''

        save_pickle(f'{base}_log.pickle', dict(
            kind='clean',
            data=self.option('data-tag'),
            footprint=self.option('footprint-tag'),
            radius=radius, thr_area_abs=thr_abs, thr_area_rel=thr_rel,
        ))
