import pandas as pd
import numpy as np

from .base import CommandBase
from .base import option

from ..footprint.reduce import reduce_peak_idx_mp
from ..util.pickle import save_pickle


class ReduceCommand(CommandBase):

    name = 'reduce'
    _type = 'peak'
    description = 'Reduce peaks'
    help = '''
'''

    options = CommandBase.options + [
        option('peak-tag', 'p', '', False, False, False, 'default'),
        option('distance', 'r', '', False, False, False, 1.6),
        option('window', 'w', '', False, False, False, 100),
        option('batch', 'b', '', False, False, False, 100),
    ]

    def _handle(self, base):
        peak_tag = self.option('peak-tag')
        peak_base = f'hotaru/peak/{peak_tag}'
        peaks = pd.read_csv(f'{peak_base}.csv', index_col=0)

        distance = float(self.option('distance'))
        window = int(self.option('window'))
        verbose = self.verbose()

        idx = reduce_peak_idx_mp(peaks, distance, window, verbose)
        peaks = peaks.iloc[idx]

        radius = self.used_radius()
        r = peaks['radius']
        cond = (radius[0] < r) & (r < radius[-1])
        removed_index = peaks.index[~cond]
        peaks[cond].to_csv(f'{base}.csv')

        save_pickle(f'{base}_log.pickle', dict(
            kind='reduce', peak=peak_tag, distance=distance,
            removed_index=removed_index,
        ))
