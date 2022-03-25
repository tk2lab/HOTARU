import pandas as pd

from .base import CommandBase
from .options import tag_options
from .options import options

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
        options['data_tag'],
        options['peak_tag'],
        options['batch'],
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
            peak=peak_tag, mask=mask,
        ))
