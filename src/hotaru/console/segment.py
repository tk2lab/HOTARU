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

    def create(self, data, prev, curr, logs, thr_dist):
        tfrecord = load_tfrecord(f'{data}-data')
        mask = load_numpy(f'{data}-mask')
        gauss, radius, shard = load_pickle(f'{prev}-filter')
        thr_out = self.status.params['thr-out']
        pos = load_numpy(f'{prev}-peak')
        score = load_numpy(f'{prev}-intensity')
        batch = self.status.params['batch']
        verbose = self.status.params['pbar']
        radius = np.array(radius)
        nr = radius.size
        writer = tf.summary.create_file_writer(logs)
        with writer.as_default():
            n = 10
            for i in range(6, n + 1):
                thr = i * thr_dist / n
                idx = reduce_peak_idx(pos, thr)
                r = radius[pos[idx, 1]]
                s = score[idx]
                tf.summary.histogram(f'radius/{curr[-3:]}', r, step=i)
                tf.summary.histogram(f'intensity/{curr[-3:]}', s, step=i)
                writer.flush()

            pos = pos[idx]
            cond = (0 < pos[:, 1]) & (pos[:, 1] < nr - 1)
            pos = pos[cond]
            r = radius[pos[:, 1]]
            s = s[cond]
            tf.summary.histogram(f'radius/{curr[-3:]}', r, step=n+1)
            tf.summary.histogram(f'intensity/{curr[-3:]}', s, step=n+1)
            writer.flush()

            segment = make_segment(
                tfrecord, mask, gauss, radius, pos, shard, batch, verbose,
            )
            nk = segment.shape[0]
            fsum = segment.sum(axis=1)
            tf.summary.histogram(f'sum_val/{curr[-3:]}', fsum, step=0)
            cond = np.ones(nk, np.bool)
            summary_segment(segment, mask, cond, gauss, thr_out, curr[-3:])
            writer.flush()
        writer.close()
        save_numpy(f'{curr}-segment', segment)
        save_numpy(f'{curr}-peak', pos[:, [0, 2, 3]])
        save_numpy(f'{curr}-radius', r)
        save_numpy(f'{curr}-intensity', s)
