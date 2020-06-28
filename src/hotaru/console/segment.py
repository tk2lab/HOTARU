import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..footprint.reduce import reduce_peak_idx
from ..footprint.make import make_segment
from ..train.summary import summary_segment
from ..util.tfrecord import load_tfrecord
from ..util.numpy import load_numpy, save_numpy
from ..util.pickle import load_pickle


class SegmentCommand(Command):

    name = 'segment'
    description = 'Make segment'
    help = '''
'''

    options = [
        _option('job-dir'),
        option('force', 'f'),
    ]

    def is_error(self, stage):
        return stage < 2

    def is_target(self, stage):
        return stage == 2

    def force_stage(self, stage):
        return 2

    def create(self, data, prev, curr, logs, thr_intensity, thr_distance):
        batch = self.status.params['batch']
        verbose = self.status.params['pbar']
        thr_out = self.status.params['thr-out']

        tfrecord = load_tfrecord(f'{data}-data')
        mask = load_numpy(f'{data}-mask')
        gauss, radius, shard = load_pickle(f'{prev}-filter')
        radius = np.array(radius)
        nr = radius.size
        p = load_numpy(f'{prev}-peak')
        r = radius[p[:, 1]]
        s = load_numpy(f'{prev}-intensity')

        writer = tf.summary.create_file_writer(logs)
        with writer.as_default():
            cond = s > thr_intensity
            p = p[cond]
            r = r[cond]
            s = s[cond]
            tf.summary.histogram(f'init/radius', r, step=0)
            tf.summary.histogram(f'init/intensity', s, step=0)
            writer.flush()

            idx = reduce_peak_idx(p, radius, thr_distance)
            p = p[idx]
            r = r[idx]
            s = s[idx]
            tf.summary.histogram(f'init/radius', r, step=1)
            tf.summary.histogram(f'init/intensity', s, step=1)
            writer.flush()

            cond = (0 < p[:, 1]) & (p[:, 1] < nr - 1)
            p = p[cond]
            r = r[cond]
            s = s[cond]
            tf.summary.histogram(f'init/radius', r, step=2)
            tf.summary.histogram(f'init/intensity', s, step=2)
            writer.flush()

            segment = make_segment(
                tfrecord, mask, gauss, radius, p, shard, batch, verbose,
            )

            nk = segment.shape[0]
            cond = np.ones(nk, np.bool)
            summary_segment(segment, mask, cond, gauss, thr_out, curr[-3:])
            fsum = segment.sum(axis=1)
            tf.summary.histogram(f'init/seg_size', fsum, step=0)
            writer.flush()
        writer.close()

        save_numpy(f'{curr}-segment', segment)
        save_numpy(f'{curr}-peak', p[:, [0, 2, 3]])
        save_numpy(f'{curr}-radius', r)
        save_numpy(f'{curr}-intensity', s)
