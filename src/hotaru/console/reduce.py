import shutil

from .base import CommandBase
from .options import tag_options
from .options import options

from ..footprint.reduce import reduce_peak_idx_mp
from ..footprint.reduce import remove_out_of_range
from ..util.csv import load_csv
from ..util.csv import save_csv
from ..util.pickle import save_pickle


class ReduceCommand(CommandBase):

    name = 'reduce'
    _type = 'peak'
    description = 'Reduce peaks'
    help = '''
'''

    options = CommandBase.options + [
        tag_options['peak_tag'],
        options['distance'],
        options['window'],
        options['batch'],
    ]

    def _handle(self, base):
        peak_tag = self.option('peak-tag') + '_find'
        peak_base = f'hotaru/peak/{peak_tag}'
        peaks = load_csv(f'{peak_base}.csv')

        distance = float(self.option('distance'))
        window = int(self.option('window'))
        verbose = self.verbose()

        idx = reduce_peak_idx_mp(peaks, distance, window, verbose)
        peaks = peaks.iloc[idx]

        radius = self.used_radius(peak_tag)
        peaks, removed_index = remove_out_of_range(peaks, radius)
        save_csv(f'{base}.csv', peaks)

        save_pickle(f'{base}_log.pickle', dict(
            kind='reduce', peak=peak_tag, distance=distance,
            radius=radius, removed_index=removed_index,
        ))
