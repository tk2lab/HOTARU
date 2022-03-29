from .base import CommandBase
from .options import options

from ..image.load import load_data
from ..image.mask import get_mask
from ..image.mask import get_mask_range
from ..image.max import calc_max
from ..image.std import calc_std
from ..image.cor import calc_cor


class StatCommand(CommandBase):

    name = 'stat'
    _type = 'data'
    _suff = '_stat'
    description = 'calc stat'
    help = '''
'''

    options = CommandBase.base_options() + [
        options['imgs-path'],
        options['mask-type'],
        options['batch'],
    ]

    def _handle(self, p):
        batch = p['batch']
        verbose = p['verbose']

        imgs = load_data(p['imgs-path'])
        nt, h, w = imgs.shape()

        mask = get_mask(p['mask-type'], h, w)
        y0, y1, x0, x1 = get_mask_range(mask)

        data = imgs.clipped_dataset(y0, y1, x0, x1)
        mask = mask[y0:y1, x0:x1]

        mmax = calc_max(data.batch(batch), nt, verbose)
        mcor = calc_cor(data.batch(batch), nt, verbose)
        mstd = calc_std(data.batch(batch), nt, verbose)

        p.update(dict(
            nt=nt, y0=y0, x0=x0, mask=mask,
            mmax=mmax, mstd=mstd, mcor=mcor,
        ))
