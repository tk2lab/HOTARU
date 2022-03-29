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

    options = CommandBase.base_options() + [
        options['data-tag'],
        tag_options['footprint-tag'],
        options['stage'],
    ] + radius_options + [
        options['thr-area-abs'],
        options['thr-area-rel'],
        options['thr-sim'],
        options['batch'],
    ]

    def log_path(self):
        tag = self.p('tag')
        stage = self.p('stage')
        if stage < 0:
            curr = ''
        else:
            curr = f'_{stage:03}'
        return f'hotaru/{self._type}/{tag}{curr}_log.pickle'

    def _handle(self, p):
        mask, nt = self.data_prop()
        radius = self.radius()

        footprint_tag = p['footprint-tag']
        stage = p['stage']
        curr = '' if stage < 0 else f'_{stage:03}'
        footprint = load_numpy(f'hotaru/footprint/{footprint_tag}{curr}_orig.npy')
        old_nk = footprint.shape[0]

        footprint, peaks = clean_footprint(
            footprint, mask, radius, p['batch'], p['verbose'],
        )

        check_accept(
            footprint, peaks, radius,
            p['thr-area-abs'], p['thr-area-rel'], p['thr-sim'],
        )

        cond = peaks['accept'] == 'yes'
        tag = p['tag']
        base = f'hotaru/footprint/{tag}{curr}'
        save_csv(f'{base}_peaks.csv', peaks)
        save_numpy(f'{base}.npy', footprint[cond])
        save_numpy(f'{base}_removed.npy', footprint[~cond])
        if stage >= 0:
            save_numpy(f'hotaru/footprint/{tag}.npy', footprint[cond])

        nk = cond.sum()
        self.line(f'ncell: {old_nk} -> {nk}', 'comment')
        if nk > 0:
            for l in str(peaks[cond]).split('\n'):
                self.line(l)
        if nk != old_nk:
            for l in str(peaks[~cond]).split('\n'):
                self.line(l)

        p.update(dict(mask=mask, nt=nt, old_nk=old_nk, nk=nk, radius=radius))
