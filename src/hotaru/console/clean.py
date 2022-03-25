import numpy as np

from .base import CommandBase
from .options import tag_options
from .options import options
from .options import radius_options

from ..footprint.clean import clean_footprint
from ..footprint.clean import to_be_removed
from ..util.numpy import load_numpy
from ..util.numpy import save_numpy
from ..util.csv import save_csv
from ..util.pickle import save_pickle


class CleanCommand(CommandBase):

    name = 'clean'
    _type = 'footprint'
    description = 'Clean segment'
    help = '''The clean command 
'''

    options = CommandBase.options + [
        options['data_tag'],
        tag_options['footprint_tag'],
    ] + radius_options + [
        options['thr_area_abs'],
        options['thr_area_rel'],
        options['batch'],
    ]

    def _handle(self, base):
        footprint_tag = self.option('footprint-tag') + '_orig'
        footprint_base = f'hotaru/footprint/{footprint_tag}'
        footprint = load_numpy(f'{footprint_base}.npy')

        mask = self.mask()
        radius = self.radius()
        batch = self.option('batch')
        verbose = self.verbose()

        footprint, peaks = clean_footprint(
            footprint, mask, radius, batch, verbose,
        )

        thr_abs = self.option('thr-area-abs')
        thr_rel = self.option('thr-area-rel')
        cond = to_be_removed(footprint, peaks, radius, thr_abs, thr_rel)
        peaks['accepted'] = False
        peaks.loc[cond, 'accepted'] = True

        old_nk = footprint.shape[0]
        nk = cond.sum()
        self.line(f'ncell: {old_nk} -> {nk}', 'comment')
        save_csv(f'{base}_peaks.csv', peaks)
        save_numpy(f'{base}.npy', footprint[cond])
        save_numpy(f'{base}_removed.npy', footprint[~cond])

        if np.any(cond):
            for l in str(peaks[cond]).split('\n'):
                self.line(l)
        if np.any(~cond):
            for l in str(peaks[~cond]).split('\n'):
                self.line(l)

        save_pickle(f'{base}_log.pickle', dict(
            kind='clean',
            data=self.option('data-tag'),
            footprint=self.option('footprint-tag'),
            mask=mask, radius=radius,
            thr_area_abs=thr_abs, thr_area_rel=thr_rel,
        ))
