import numpy as np

from .base import CommandBase
from .options import options
from .options import tag_options
from .options import radius_options

from ..footprint.clean import clean_footprint
from ..footprint.clean import check_accept
from ..util.numpy import load_numpy
from ..util.csv import load_csv
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

        stage = p['stage']
        curr = '' if stage < 0 else f'_{stage:03}'
        footprint = load_numpy(f'hotaru/footprint/{tag}{curr}_orig.npy')
        index = load_pickle(f'hotaru/footprint/{tag}{curr}_orig_log.npy')
        old_nk = footprint.shape[0]

        cond = modify_footprint(footprint)

        footprint, peaks = clean_footprint(
            footprint[cond], index[cond].index,
            mask, radius, p['batch'], p['verbose'],
        )
        peaks.loc[index[~cond], 'accept'] = 'no_seg'

        idx = np.argsort(peaks['firmness'].values)[::-1]
        foooprint = footprint[idx]
        peaks = peaks.iloc[idx].copy()

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
            save_csv(f'hotaru/footprint/{tag}_peaks.csv', peaks)
            save_numpy(f'hotaru/footprint/{tag}.npy', footprint[cond])

        nk = cond.sum()
        #print(peaks.query('accept=="yes"'))
        #print(peaks.query('accept!="yes"'))
        if nk > 0:
            for l in str(peaks[cond]).split('\n'):
                self.line(l)
        if nk != old_nk:
            for l in str(peaks[~cond]).split('\n'):
                self.line(l)
        sim = peaks.query('accept=="yes"')['sim'].values
        print('sim', np.sort(sim[sim > 0]))
        self.line(f'ncell: {old_nk} -> {nk}', 'comment')
        p.update(dict(mask=mask, nt=nt, old_nk=old_nk, nk=nk, radius=radius))
