import os

import tensorflow as tf
import numpy as np

from .base import Command, option
from ..data.load import get_shape, load_data
from ..data.mask import get_mask, get_mask_range, make_maskfile, load_maskfile
from ..data.std import calc_std
from ..data.dataset import normalized, masked
from ..data.tfrecord import make_tfrecord


class DataCommand(Command):

    description = 'Create TFRecord'

    name = 'data'
    options = [
        option('job-dir', flag=False, value_required=False),
        option('imgs-file', flag=False, value_required=False),
        option('mask-type', flag=False, value_required=False),
        option('batch', flag=False, default=100),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()

        default_imgs_file = 'imgs.tif'
        default_mask_type = '0.pad'

        imgs_file = self.status_value('imgs-file', default_imgs_file)
        mask_type = self.status_value('mask-type', default_mask_type)
        batch = int(self.option('batch'))

        imgs_file_path = os.path.join(self.application.job_dir, imgs_file)
        data_file = os.path.join(self.work_dir, 'data.tfrecord')
        mask_file = os.path.join(self.work_dir, 'mask.npy')
        avgt_file = os.path.join(self.work_dir, 'avgt.npy')
        avgx_file = os.path.join(self.work_dir, 'avgx.npy')

        if self.option('force') or not tf.io.gfile.exists(mask_file):
            self.line('data')
            nt, h, w = get_shape(imgs_file_path)
            self.status['root']['nt'] = nt

            mask = get_mask(mask_type, h, w, self.application.job_dir)
            y0, y1, x0, x1 = get_mask_range(mask)
            mask = mask[y0:y1, x0:x1]
            make_maskfile(mask_file, mask)
            self.status['root']['rect'] = y0, y1, x0, x1
            self.status['root']['nx'] = np.count_nonzero(mask)
            self.status['root']['h'] = y1 - y0
            self.status['root']['w'] = x1 - x0

            avgt, avgx, std = calc_std(self.imgs, mask, nt, batch)
            make_maskfile(avgt_file, avgt.numpy())
            make_maskfile(avgx_file, avgx.numpy())
            self.status['root']['std'] = std.numpy()

            prog = tf.keras.utils.Progbar(nt)
            masked_data = masked(self.normalized_imgs, mask)
            make_tfrecord(data_file, masked_data, prog=prog)

            self.save_status()

    def status_value(self, name, default_val=None, dtype=str):
        current_val = self.status['root'].get(name, None)
        val = self.option(name)
        if current_val is None:
            val = val or default_val
            if val is None:
                raise RuntimeError(f'config missing: {name}')
            if isinstance(val, list):
                val = tuple(dtype(v) for v in val)
            else:
                val = dtype(val)
            self.status['root'][name] = val
            self.save_status()
        elif val and val != current_val:
            raise RuntimeError(f'config mismatch: {name}')
        return self.status['root'][name]
