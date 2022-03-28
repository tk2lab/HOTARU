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
    _suff = '_000'
    description = 'Make initial segment'
    help = '''
'''

    options = CommandBase.base_options() + [
        options['data-tag'],
        options['peak-tag'],
    ] + radius_options + [
        options['distance'],
        options['window'],
        options['batch'],
    ]

    def _handle(self, p):
        data = self.data()
        mask, nt, avgx = self.data_prop(avgx=True)
        radius = self.radius()

        peak_tag = p['peak-tag']
        peaks = load_csv(f'hotaru/peak/{peak_tag}.csv')

        idx = reduce_peak_idx_mp(
            peaks, p['distance'], p['window'], p['verbose'],
        )
        tag = p['tag']
        peaks = label_out_of_range(peaks.loc[idx], radius[0], radius[-1])
        save_csv(f'hotaru/footprint/{tag}_000_peaks.csv', peaks)

        peaks = peaks.query('accept == "yes"')
        segment, ok_mask = make_segment(
            data, mask, avgx, peaks, p['batch'], p['verbose'],
        )
        tag = p['tag']
        save_numpy(f'hotaru/footprint/{tag}_000.npy', segment)
        save_numpy(f'hotaru/footprint/{tag}.npy', segment)

        p.update(dict(radius=radius, mask=mask, nt=nt))
