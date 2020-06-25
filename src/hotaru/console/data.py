import os

import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..image.load import get_shape, load_data
from ..image.mask import get_mask, get_mask_range
from ..image.std import calc_std
from ..util.dataset import normalized, masked
from ..util.tfrecord import save_tfrecord
from ..util.pickle import save_pickle
from ..util.numpy import save_numpy


class DataCommand(Command):

    name = 'data'
    description = 'Create TFRecord'
    help = '''
'''

    options = [
        _option('job-dir'),
        option('force', 'f'),
    ]

    def is_error(self, stage):
        return False

    def is_target(self, stage):
        return stage == 0

    def force_stage(self, stage):
        return 0

    def create(self, data, prev, curr, log, imgs_file, mask_type):
        def _gen():
            for x in imgs:
                yield tf.convert_to_tensor(wrap(x)[y0:y1, x0:x1], tf.float32)

        imgs_file = os.path.join(self.application.job_dir, imgs_file)
        batch = self.status.params['batch']
        verbose = self.status.params['pbar']

        imgs, wrap = load_data(imgs_file)
        nt, h, w = get_shape(imgs_file)
        mask = get_mask(mask_type, h, w, self.application.job_dir)

        y0, y1, x0, x1 = get_mask_range(mask)
        data = tf.data.Dataset.from_generator(_gen, tf.float32)
        mask = mask[y0:y1, x0:x1]

        avgt, avgx, std = calc_std(data.batch(batch), mask, nt, verbose)
        nx = np.count_nonzero(mask)
        stat = nx, nt, h, w, y0, x0, std

        normalized_data = normalized(data, avgt, avgx, std)
        masked_data = masked(normalized_data, mask)

        save_tfrecord(f'{curr}-data', masked_data, nt, verbose)
        save_numpy(f'{curr}-mask', mask)
        save_numpy(f'{curr}-avgt', avgt)
        save_numpy(f'{curr}-avgx', avgx)
        save_pickle(f'{curr}-stat', stat)
