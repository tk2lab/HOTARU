import pandas as pd

from .base import CommandBase
from .base import option

from ..footprint.make import make_segment
from ..util.numpy import save_numpy
from ..util.pickle import save_pickle


class InitCommand(CommandBase):

    name = 'init'
    _type = 'footprint'
    description = 'Make initial segment'
    help = '''
'''

    options = CommandBase.options + [
        option('data-tag', 'd', '', False, False, False, 'default'),
        option('peak-tag', 'p', '', False, False, False, 'default'),
        option('batch', 'b', '', False, False, False, 100),
    ]

    def _handle(self, base):
        data, mask, nt = self.data()

        peak_tag = self.option('peak-tag')
        peak_base = f'hotaru/peak/{peak_tag}'
        peaks = pd.read_csv(f'{peak_base}.csv')

        batch = int(self.option('batch'))
        verbose = self.verbose()
        segment, ok_mask = make_segment(data, mask, peaks, batch, verbose)
        save_numpy(f'{base}.npy', segment)

        save_pickle(f'{base}_log.pickle', dict(
            peak=peak_tag,
        ))
