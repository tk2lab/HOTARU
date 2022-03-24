import pandas as pd

from .base import CommandBase
from .radius import RadiusMixin
from .base import option

from ..footprint.find import find_peak
from ..util.pickle import save_pickle


class FindCommand(CommandBase, RadiusMixin):

    name = 'find'
    _type = 'peak'
    description = 'Find peaks'
    help = '''
'''

    options = CommandBase.options + [
        option('data-tag', 'd', '', False, False, False, 'default'),
        option('shard', 's', '', False, False, False, 1),
    ] + RadiusMixin._options + [
        option('batch', 'b', '', False, False, False, 100),
    ]

    def _handle(self, base):
        data, mask, nt = self.data()

        radius = self.radius()
        shard = int(self.option('shard'))
        batch = int(self.option('batch'))
        verbose = self.verbose()

        find_peak(
            data, mask, radius, shard, batch, nt, verbose,
        ).to_csv(f'{base}.csv')

        save_pickle(f'{base}_log.pickle', dict(
            kind='find',
            data=self.option('data-tag'), shard=shard, radius=radius,
        ))
