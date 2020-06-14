import os

import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

from ..image.load import get_shape, load_data
from ..image.mask import get_mask, get_mask_range
from ..image.std import calc_std
from ..util.tfrecord import make_tfrecord
from ..util.dataset import normalized, masked
from ..util.npy import save_numpy, load_numpy
from .base import Command, option


class DataCommand(Command):

    description = 'Create TFRecord'

    name = 'data'
    options = [
        option('job-dir', flag=False, value_required=False),
        option('force', 'f', 'overwrite previous result'),
    ]

    @property
    def imgs(self):

        def _gen():
            for x in imgs:
                yield tf.convert_to_tensor(wrap(x)[y0:y1, x0:x1], tf.float32)

        imgs_file = self.status['root']['imgs-file']
        imgs_file = os.path.join(self.application.job_dir, imgs_file)
        imgs, wrap = load_data(imgs_file)
        y0, y1, x0, x1 = self.status['root']['rect']
        return tf.data.Dataset.from_generator(_gen, tf.float32)

    @property
    def normalized_imgs(self):
        avgt = load_numpy(os.path.join(self.work_dir, 'avgt'))
        avgx = load_numpy(os.path.join(self.work_dir, 'avgx'))
        std = self.status['root']['std']
        return normalized(self.imgs, avgt, avgx, std)

    def handle(self):
        self.set_job_dir()

        imgs_file = self.status['root']['imgs-file']
        mask_type = self.status['root']['mask-type']
        batch = self.status['root']['batch']

        imgs_file_path = os.path.join(self.application.job_dir, imgs_file)
        data_file = os.path.join(self.work_dir, 'data.tfrecord')
        mask_base = os.path.join(self.work_dir, 'mask')
        avgt_base = os.path.join(self.work_dir, 'avgt')
        avgx_base = os.path.join(self.work_dir, 'avgx')

        if self.option('force') or not tf.io.gfile.exists(data_file):
            self.line('data')
            nt, h, w = get_shape(imgs_file_path)
            self.status['root']['nt'] = nt

            mask = get_mask(mask_type, h, w, self.application.job_dir)
            y0, y1, x0, x1 = get_mask_range(mask)
            mask = mask[y0:y1, x0:x1]
            self.status['root']['rect'] = y0, y1, x0, x1
            self.status['root']['nx'] = np.count_nonzero(mask)
            self.status['root']['h'] = y1 - y0
            self.status['root']['w'] = x1 - x0

            prog = tf.keras.utils.Progbar(nt)
            mask = K.constant(mask, tf.bool)
            avgt, avgx, std = calc_std(self.imgs.batch(batch), mask, prog)
            save_numpy(avgt_base, avgt)
            save_numpy(avgx_base, avgx)
            self.status['root']['std'] = std

            prog = tf.keras.utils.Progbar(nt)
            masked_data = masked(self.normalized_imgs, mask)
            make_tfrecord(data_file, masked_data, prog=prog)
            save_numpy(mask_base, mask)

        self.save_status()
