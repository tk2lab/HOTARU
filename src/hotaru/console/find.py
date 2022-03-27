from .base import CommandBase
from .options import options
from .options import radius_options

from ..footprint.find import find_peak
from ..util.csv import save_csv


class FindCommand(CommandBase):

    name = 'find'
    _type = 'peak'
    description = 'Find peaks'
    help = '''
'''

    options = CommandBase.base_options() + [
        options['data-tag'],
        options['shard'],
    ] + radius_options + [
        options['batch'],
    ]

    def _handle(self, base, p):
        data = self.data()
        mask, nt, avgx = self.data_prop(avgx=True)
        radius = self.radius()

        peaks = find_peak(
            data, mask, avgx, radius, p['shard'], p['batch'], nt, p['verbose'],
        )
        save_csv(f'{base}.csv', peaks)

        p.update(dict(mask=mask, nt=nt, radius=radius))
