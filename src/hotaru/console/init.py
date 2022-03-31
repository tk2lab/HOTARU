import numpy as np
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
        tag = p['tag']

        data = self.data()
        mask, nt, avgx = self.data_prop(avgx=True)
        radius = self.radius()
        avgx = np.zeros_like(avgx)

        peak_tag = p['peak-tag']
        peaks = load_csv(f'hotaru/peak/{peak_tag}.csv')

        idx = reduce_peak_idx_mp(
            peaks, p['distance'], p['window'], p['verbose'],
        )
        peaks = peaks.iloc[idx]
        label_out_of_range(peaks, radius[0], radius[-1])

        n0 = peaks.shape[0]
        n1 = (peaks['accept'] == 'yes').values.sum()
        segment, ng_index = make_segment(
            data, mask, avgx, peaks.query('accept=="yes"'), p['batch'], p['verbose'],
        )
        peaks.loc[ng_index, 'accept'] = 'no_seg'
        n2 = (peaks['accept'] == 'yes').values.sum()
        n3 = segment.shape[0]

        tag = p['tag']
        save_numpy(f'hotaru/footprint/{tag}_000.npy', segment)

        print(peaks.query('accept=="yes"'))
        print(peaks.query('accept!="yes"'))
        self.line(f'{n0} -> {n1} -> {n2}')
        p.update(dict(radius=radius, peaks=peaks))
