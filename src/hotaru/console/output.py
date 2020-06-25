import os

import tensorflow as tf
import numpy as np

from .base import Command, _option
from ..image.filter.gaussian import gaussian
from ..train.dynamics import SpikeToCalcium
from ..util.numpy import load_numpy
from ..util.pickle import load_pickle
from ..util.tiff import save_tiff
from ..util.csv import save_csv


class OutputCommand(Command):

    name = 'output'
    description = 'Output results'
    help = '''
'''

    options = [
        _option('job-dir'),
    ]

    def handle(self):
        self.set_job_dir()

        name = self.status.params['name']
        thr_out = self.status.params['thr-out']
        out_dir = os.path.join(self.application.job_dir, 'outs')
        out_base = os.path.join(out_dir, name)
        tf.io.gfile.makedirs(out_dir)

        history = self.status.history.get(name, ())
        stage = 3 * ((len(history) - 0) // 3) + 0
        data = self.status.find_saved(history[:1])
        spike = self.status.find_saved(history[:stage])
        segment = self.status.find_saved(history[:stage-1])
        data = os.path.join(self.work_dir, data, '000')
        spike = os.path.join(self.work_dir, spike, f'{stage:03d}')
        segment = os.path.join(self.work_dir, segment, f'{stage-1:03d}')

        mask0 = load_numpy(f'{data}-mask')
        h0, w0 = mask0.shape
        nx, nt, h, w, y0, x0, std = load_pickle(f'{data}-stat')
        mask = np.zeros((h, w), np.bool)
        mask[y0:y0+h0, x0:x0+w0] = mask0

        segment = load_numpy(f'{segment}-segment')
        spike = load_numpy(f'{spike}-spike')
        nk, nu = spike.shape

        out = np.zeros((nk, h, w), np.float32)
        out[:, mask] = segment
        out = gaussian(out, self.status.params['gauss']).numpy()
        out -= out.min(axis=(1, 2), keepdims=True)
        out /= out.max(axis=(1, 2), keepdims=True)
        out -= thr_out
        out[out < 0.0] = 0.0
        out /= (1 - thr_out)

        hz = self.status.params['hz']
        spike_to_calcium = SpikeToCalcium()
        spike_to_calcium.set_double_exp(*self.status.tau)
        calcium = spike_to_calcium(spike).numpy()
        pad = spike_to_calcium.pad
        time = (np.arange(nu) - pad) / hz
        spike /= spike.max(axis=1, keepdims=True)
        spike = np.concatenate([time[:, None], spike.T], axis=1)
        time = time[pad:]
        calcium = np.concatenate([time[:, None], calcium.T], axis=1)
        h = ','.join(['time'] + [f'id{k:04d}' for k in range(nk)])

        save_tiff(f'{out_base}-cell', out)
        save_csv(f'{out_base}-spike', spike, '%.3f', header=h)
        save_csv(f'{out_base}-calcium', calcium, '%.3f', header=h)
