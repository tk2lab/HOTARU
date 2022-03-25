from .base import CommandBase
from .options import options

from ..image.load import load_data
from ..image.mask import get_mask
from ..image.mask import get_mask_range
from ..image.std import calc_std
from ..image.max import calc_max
from ..image.cor import calc_cor
from ..util.dataset import normalized
from ..util.dataset import masked
from ..util.tfrecord import save_tfrecord
from ..util.pickle import save_pickle


class DataCommand(CommandBase):

    name = 'data'
    _type = 'data'
    description = 'Create TFRecord'
    help = '''
'''

    options = CommandBase.options + [
        options['imgs_path'],
        options['mask_type'],
        options['batch'],
    ]

    def _handle(self, base):
        imgs_path = self.option('imgs-path')
        imgs = load_data(imgs_path)
        nt, h, w = imgs.shape()

        mask_type = self.option('mask-type')
        mask = get_mask(mask_type, h, w)
        y0, y1, x0, x1 = get_mask_range(mask)

        data = imgs.clipped_dataset(y0, y1, x0, x1)
        mask = mask[y0:y1, x0:x1]

        batch = int(self.option('batch'))
        verbose = self.verbose()
        istd, avgt, avgx = calc_std(data.batch(batch), mask, nt, verbose)
        #imax = calc_max(data.batch(batch), nt, verbose)
        #icor = calc_cor(data.batch(batch), nt, verbose)

        data_path = f'{base}.tfrecord'
        normalized_data = normalized(data, istd, avgt, avgx)
        masked_data = masked(normalized_data, mask)
        save_tfrecord(data_path, masked_data, nt, verbose)

        log_path = f'{base}_log.pickle'
        save_pickle(log_path, dict(
            imgs=imgs_path, nt=nt, y0=y0, x0=x0, mask=mask,
            std=istd, avgt=avgt, avgx=avgx, #max=imax, cor=icor,
        ))
