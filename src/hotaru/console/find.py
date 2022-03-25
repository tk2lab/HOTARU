from .base import CommandBase
from .options import tag_options
from .options import options
from .options import radius_options

from ..footprint.find import find_peak
from ..util.csv import save_csv
from ..util.pickle import save_pickle


class FindCommand(CommandBase):

    name = 'find'
    _suff = '_find'
    _type = 'peak'
    description = 'Find peaks'
    help = '''
'''

    options = CommandBase.options + [
        tag_options['data_tag'],
        options['shard'],
    ] + radius_options + [
        options['batch'],
    ]

    def _handle(self, base):
        data, mask, nt = self.data()

        radius = self.radius()
        shard = int(self.option('shard'))
        batch = int(self.option('batch'))
        verbose = self.verbose()

        peaks = find_peak(
            data, mask, radius, shard, batch, nt, verbose,
        )
        save_csv(f'{base}.csv', peaks)

        save_pickle(f'{base}_log.pickle', dict(
            kind='find',
            data=self.option('data-tag'), shard=shard, radius=radius,
        ))
