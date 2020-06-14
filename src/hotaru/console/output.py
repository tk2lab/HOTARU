import os

import tensorflow as tf
import pandas as pd
import numpy as np
import tifffile

from .base import Command, _option
from ..image.filter.gaussian import gaussian


class OutputCommand(Command):

    description = 'Output results'

    name = 'output'
    options = [
        _option('job-dir'),
    ]

    def handle(self):
        self.set_job_dir()

        out_dir = os.path.join(self.application.job_dir, 'out')
        tf.io.gfile.makedirs(out_dir)

        footprint = self.clean
        mask = self.mask
        nk = footprint.shape[0]
        h, w = mask.shape
        out = np.zeros((nk, h, w), np.float32)
        out[:, mask] = footprint
        out = gaussian(out, 2.0).numpy()
        out -= out.min(axis=(1,2), keepdims=True)
        out /= out.max(axis=(1,2), keepdims=True)
        tifffile.imwrite(os.path.join(out_dir, 'cell.tif'), out)

        spike = self.spike
        nk, nu = spike.shape
        pad = self.model.variance.spike_to_calcium.pad
        hz = self.status['root']['hz']
        calcium = self.model.variance.spike_to_calcium(spike)
        time = (np.arange(nu) - pad) / hz
        df = pd.DataFrame(dict(time=time))
        for k in range(nk):
            df[f'id{k:04d}'] = spike[k] / spike[k].max()
        df.to_csv(os.path.join(out_dir, 'spike.csv'), float_format='%.3f')
        time = time[pad:]
        df = pd.DataFrame(dict(time=time))
        for k in range(nk):
            df[f'id{k:04d}'] = calcium[k]
        df.to_csv(os.path.join(out_dir, 'calcium.csv'), float_format='%.3f')
