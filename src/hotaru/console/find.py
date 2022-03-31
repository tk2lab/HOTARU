import numpy as np

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

    def _handle(self, p):
        data = self.data()
        mask, nt, avgx = self.data_prop(avgx=True)
        radius = self.radius()
        avgx = np.zeros_like(avgx)

        peaks = find_peak(
            data, mask, avgx, radius, p['shard'], p['batch'], nt, p['verbose'],
        )
        tag = p['tag']
        save_csv(f'hotaru/peak/{tag}.csv', peaks)

        p.update(dict(mask=mask, nt=nt, radius=radius))
