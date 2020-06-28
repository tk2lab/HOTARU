import os

import tensorflow as tf
import numpy as np

from .base import Command, option, _option
from ..footprint.clean import clean_footprint
from ..image.filter.gaussian import gaussian
from ..train.summary import normalized_and_sort
from ..train.summary import summary_segment
from ..util.numpy import load_numpy, save_numpy


class CleanCommand(Command):

    name = 'clean'
    description = 'Clean segment'
    help = '''The clean command 
'''

    options = [
        _option('job-dir', 'j', ''),
        option('force', 'f', ''),
    ]

    def is_error(self, stage):
        return stage < 2

    def is_target(self, stage):
        return stage % 3 == 2

    def force_stage(self, stage):
        return 3 * ((stage - 2) // 3) + 2

    def create(self, data, prev, curr, logs, gauss, radius,
               thr_firmness, thr_sim_area, thr_similarity):
        footprint = load_numpy(f'{prev}-footprint')
        mask = load_numpy(f'{data}-mask')
        batch = self.status.params['batch']
        verbose = self.status.params['pbar']
        thr_out = self.status.params['thr-out']
        radius = np.array(radius)
        nr = radius.size
        segment, pos, firmness = clean_footprint(
            footprint, mask, gauss, radius, batch, verbose,
        )

        idx = np.argsort(firmness)[::-1]
        segment = segment[idx]
        pos = pos[idx]
        fir = firmness[idx]
        rad = radius[pos[:, 0]]

        old_nk = segment.shape[0]
        if thr_sim_area > 0.0:
            h, w = mask.shape
            seg = np.zeros((old_nk, h, w), np.float32)
            seg[:, mask] = segment
            seg = gaussian(seg, gauss).numpy()
            seg -= seg.min(axis=(1, 2), keepdims=True)
            mag = seg.max(axis=(1, 2))
            cond = mag > 0.0
            seg[cond] /= mag[cond, None, None]
            seg[~cond] = 1.0
            seg = seg > thr_sim_area
            cor = np.zeros((old_nk,))
            for j in np.arange(old_nk)[::-1]:
                cj = 0.0
                for i in np.arange(j)[::-1]:
                    ni = np.count_nonzero(seg[i])
                    nij = np.count_nonzero(seg[i] & seg[j])
                    cij = nij / ni
                    if cij > cj:
                        cj = cij
                cor[j] = cj
        else:
            scale = np.sqrt((segment ** 2).sum(axis=1))
            seg = segment / scale[:, None]
            cor = np.zeros((old_nk,))
            for j in np.arange(old_nk)[::-1]:
                cj = 0.0
                for i in np.arange(j)[::-1]:
                    cij = np.dot(seg[i], seg[j])
                    if cij > cj:
                        cj = cij
                cor[j] = cj

        logs = os.path.join(logs, 'clean')
        writer = tf.summary.create_file_writer(logs)
        with writer.as_default():
            fsum = segment.sum(axis=1)
            tf.summary.histogram(f'seg_radius/{curr[-3:]}', rad, step=0)
            tf.summary.histogram(f'seg_firmness/{curr[-3:]}', fir, step=0)
            tf.summary.histogram(f'seg_correlation/{curr[-3:]}', cor, step=0)
            tf.summary.histogram(f'seg_size/{curr[-3:]}', fsum, step=0)
            writer.flush()

            cond = fir > thr_firmness
            cond &= cor < thr_similarity
            cond &= (0 < pos[:, 0]) & (pos[:, 0] < nr - 1)
            idx = np.where(~cond)[0]
            for i in idx:
                print(i, fir[i], cor[i], pos[i, 0])
            summary_segment(segment, mask, cond, gauss, thr_out, curr[-3:])
            writer.flush()

            segment = segment[cond]
            pos = pos[cond]
            rad = rad[cond]
            fir = fir[cond]
            cor = cor[cond]

            fsum = segment.sum(axis=1)
            tf.summary.histogram(f'seg_radius/{curr[-3:]}', rad, step=1)
            tf.summary.histogram(f'seg_firmness/{curr[-3:]}', fir, step=1)
            tf.summary.histogram(f'seg_correlation/{curr[-3:]}', cor, step=1)
            tf.summary.histogram(f'seg_size/{curr[-3:]}', fsum, step=1)
            writer.flush()
        writer.close()

        nk = segment.shape[0]
        self.line(f'ncell: {old_nk} -> {nk}', 'comment')

        save_numpy(f'{curr}-segment', segment)
        save_numpy(f'{curr}-pos', pos[:, 1:])
        save_numpy(f'{curr}-radius', rad)
        save_numpy(f'{curr}-firmness', fir)
