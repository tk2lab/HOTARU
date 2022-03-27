from .base import CommandBase
from .options import options

from ..image.load import load_data
from ..image.mask import get_mask
from ..image.mask import get_mask_range
from ..image.stats import calc_stats
from ..util.dataset import normalized
from ..util.dataset import masked
from ..util.tfrecord import save_tfrecord


class DataCommand(CommandBase):

    name = 'data'
    _type = 'data'
    description = 'Create TFRecord'
    help = '''
'''

    options = CommandBase.options + [
        options['imgs-path'],
        options['mask-type'],
        options['batch'],
    ]

    def _handle(self, base, p):
        batch = p['batch']
        verbose = p['verbose']

        imgs = load_data(p['imgs-path'])
        nt, h, w = imgs.shape()

        mask = get_mask(p['mask-type'], h, w)
        y0, y1, x0, x1 = get_mask_range(mask)

        data = imgs.clipped_dataset(y0, y1, x0, x1)
        mask = mask[y0:y1, x0:x1]

        stats = calc_stats(data.batch(batch), mask, nt, verbose)
        smin, smax, sstd, avgt, avgx = stats

        data_path = f'{base}.tfrecord'
        normalized_data = normalized(data, sstd, avgt, avgx)
        masked_data = masked(normalized_data, mask)
        save_tfrecord(data_path, masked_data, nt, p['verbose'])

        p.update(dict(
            nt=nt, y0=y0, x0=x0, mask=mask,
            smin=smin, smax=smax, sstd=sstd, avgt=avgt, avgx=avgx,
        ))
