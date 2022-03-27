import numpy as np

from .base import CommandBase
from .options import options
from .options import tag_options
from .options import radius_options

from ..footprint.clean import clean_footprint
from ..footprint.clean import check_accept
from ..util.numpy import load_numpy
from ..util.numpy import save_numpy
from ..util.csv import save_csv


class CleanCommand(CommandBase):

    name = 'clean'
    _type = 'footprint'
    description = 'Clean segment'
    help = '''The clean command 
'''

    options = CommandBase.options + [
        options['data-tag'],
        tag_options['footprint-tag'],
    ] + radius_options + [
        options['thr-area-abs'],
        options['thr-area-rel'],
        options['batch'],
    ]

    def _handle(self, base, p):
        mask, nt = self.data_prop()
        radius = self.radius()

        footprint_tag = p['footprint-tag'] + '_orig'
        footprint = load_numpy(f'hotaru/footprint/{footprint_tag}.npy')

        footprint, peaks = clean_footprint(
            footprint, mask, radius, p['batch'], p['verbose'],
        )

        cond = check_accept(
            footprint, peaks, radius,
            p['thr-area-abs'], p['thr-area-rel']
        )
        peaks['accept'] = np.where(cond, 'yes', 'no')
        save_csv(f'{base}_peaks.csv', peaks)
        save_numpy(f'{base}.npy', footprint[cond])
        save_numpy(f'{base}_removed.npy', footprint[~cond])

        old_nk = footprint.shape[0]
        nk = cond.sum()
        self.line(f'ncell: {old_nk} -> {nk}', 'comment')
        if nk > 0:
            for l in str(peaks[cond]).split('\n'):
                self.line(l)
        if nk != old_nk:
            for l in str(peaks[~cond]).split('\n'):
                self.line(l)

        p.update(dict(mask=mask, nt=nt, old_nk=old_nk, nk=nk, radius=radius))
