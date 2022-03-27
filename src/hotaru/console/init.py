import pandas as pd

from .base import CommandBase
from .options import options
from .options import radius_options

from ..util.csv import load_csv
from ..footprint.reduce import reduce_peak_idx_mp
from ..footprint.reduce import label_out_of_range
from ..footprint.make import make_segment
from ..util.csv import save_csv
from ..util.numpy import save_numpy


class InitCommand(CommandBase):

    name = 'init'
    _type = 'footprint'
    description = 'Make initial segment'
    help = '''
'''

    options = CommandBase.options + [
        options['data-tag'],
        options['peak-tag'],
    ] + radius_options + [
        options['distance'],
        options['window'],
        options['batch'],
    ]

    def _handle(self, base, p):
        data = self.data()
        mask, nt = self.data_prop()
        radius = self.radius()

        peak_tag = p['peak-tag']
        peaks = load_csv(f'hotaru/peak/{peak_tag}.csv')

        idx = reduce_peak_idx_mp(
            peaks, p['distance'], p['window'], p['verbose'],
        )
        peaks = label_out_of_range(peaks.loc[idx], radius)
        save_csv(f'{base}_peaks.csv', peaks)

        peaks = peaks.query('accept == "yes"')
        segment, ok_mask = make_segment(
            data, mask, peaks, p['batch'], p['verbose'],
        )
        save_numpy(f'{base}.npy', segment)

        p.update(dict(radius=radius, mask=mask, nt=nt))
