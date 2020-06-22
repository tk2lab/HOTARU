import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..footprint.clean import clean_footprint
from ..train.summary import normalized_and_sort
from ..train.summary import summary_footprint
from ..util.numpy import load_numpy, save_numpy


class CleanCommand(Command):

    description = 'Clean segment'

    name = 'clean'
    options = [
        _option('job-dir'),
        option('force', 'f'),
    ]

    def is_error(self, stage):
        return stage < 2

    def is_target(self, stage):
        return stage % 3 == 2

    def force_stage(self, stage):
        return 3 * ((stage - 2) // 3) + 2

    def create(self, data, prev, curr, logs, gauss, radius, thr_firmness):
        footprint = load_numpy(f'{prev}-footprint')
        mask = load_numpy(f'{data}-mask')
        batch = self.status.params['batch']
        radius = np.array(radius)
        nr = radius.size
        segment, pos, firmness = clean_footprint(
           footprint, mask, gauss, radius, batch,
        )

        idx = np.argsort(firmness)[::-1]
        segment = segment[idx]
        pos = pos[idx]
        f = firmness[idx]
        r = radius[pos[:, 0]]

        old_nk = segment.shape[0]
        for k in range(old_nk):
            print(r[k], f[k])

        writer = tf.summary.create_file_writer(logs)
        with writer.as_default():
            tf.summary.histogram(f'radius/{curr[-3:]}', r, step=0)
            tf.summary.histogram(f'firmness/{curr[-3:]}', f, step=0)
            writer.flush()

            fsum = segment.sum(axis=1)
            tf.summary.histogram(f'sum_val/{curr[-3:]}', fsum, step=0)
            summary_footprint(segment, mask, curr[-3:])
            writer.flush()

            cond = firmness > thr_firmness
            cond &= (0 < pos[:, 0]) & (pos[:, 0] < nr - 1)
            segment = segment[cond]
            pos = pos[cond]
            r = r[cond]
            f = f[cond]
            tf.summary.histogram(f'radius/{curr[-3:]}', r, step=1)
            tf.summary.histogram(f'firmness/{curr[-3:]}', f, step=1)
            writer.flush()
        writer.close()

        nk = segment.shape[0]
        self.line(f'ncell: {old_nk} -> {nk}', 'comment')

        save_numpy(f'{curr}-segment', segment)
        save_numpy(f'{curr}-pos', pos[:, 1:])
        save_numpy(f'{curr}-radius', r)
        save_numpy(f'{curr}-firmness', f)
