import os

import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..image.load import get_shape, load_data
from ..image.mask import get_mask, get_mask_range
from ..image.std import calc_std
from ..util.tfrecord import make_tfrecord
from ..util.dataset import normalized, masked
from ..util.npy import save_numpy, load_numpy


class DataCommand(Command):

    description = 'Create TFRecord'

    name = 'data'
    options = [
        _option('job-dir'),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):

        def _gen():
            for x in imgs:
                yield tf.convert_to_tensor(wrap(x)[y0:y1, x0:x1], tf.float32)

        self.set_job_dir()

        status = self.status['root']
        if self.option('force') or 'nt' not in status:
            self.line('<info>data</info>')

            if 'imgs-file' not in status:
                self.line_error('not configured!: please run `hotaru config`')

            imgs_file = status['imgs-file']
            mask_type = status['mask-type']
            batch = status['batch']

            imgs_file = os.path.join(self.application.job_dir, imgs_file)
            data_file = os.path.join(self.work_dir, 'data.tfrecord')
            mask_base = os.path.join(self.work_dir, 'mask')
            avgt_base = os.path.join(self.work_dir, 'avgt')
            avgx_base = os.path.join(self.work_dir, 'avgx')

            imgs, wrap = load_data(imgs_file)
            nt, h, w = get_shape(imgs_file)
            mask = get_mask(mask_type, h, w, self.application.job_dir)

            y0, y1, x0, x1 = get_mask_range(mask)
            data = tf.data.Dataset.from_generator(_gen, tf.float32)
            mask = mask[y0:y1, x0:x1]

            prog = tf.keras.utils.Progbar(nt)
            avgt, avgx, std = calc_std(data.batch(batch), mask, prog)
            normalized_data = normalized(data, avgt, avgx, std)
            masked_data = masked(normalized_data, mask)

            prog = tf.keras.utils.Progbar(nt)
            make_tfrecord(data_file, masked_data, prog=prog)
            save_numpy(avgt_base, avgt)
            save_numpy(avgx_base, avgx)
            save_numpy(mask_base, mask)

            status['nt'] = nt
            status['h'] = y1 - y0
            status['w'] = x1 - x0
            status['nx'] = np.count_nonzero(mask)
            status['margin'] = y0, x0, h - y1, w - x1
            status['std'] = std
            self.save_status()
