import datetime
import os
import io

import tensorflow as tf
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

from .base import Command, option, _option
from ..footprint.find import find_peak
from ..footprint.reduce import reduce_peak_idx, calc_distance
from ..util.dataset import unmasked
from ..util.tfrecord import moving_average_imgs
from ..util.numpy import load_numpy, save_numpy
from ..util.pickle import load_pickle, save_pickle


def draw(s, d, r):
    '''
    fig, ax = plt.subplots(2, 1, sharex='all', figsize=(8, 10), dpi=72)

    ax[0].axvline(s[0])
    sc = ax[0].scatter(
        s[1:], d[1:], s=r[1:]**2, edgecolor='r', c='none',
    )
    ax[0].set_ylabel('distance')
    ax[0].set_yscale('log')

    kde = st.gaussian_kde(s)
    l = np.linspace(0, s.max(), 101)
    ax[1].plot(l, kde(l))
    ax[1].set_xlabel('intensity')
    ax[1].set_ylabel('probability')
    '''
    s = np.log10(s[1:])
    d = np.log10(d[1:])
    jp = sns.JointGrid(s, d)
    jp.set_axis_labels('intensity', 'distance')

    #s, d = s[d > 0.0], d[s > np.log10(0.5)]

    jp.ax_marg_x.hist(s, density=True, bins=100)
    l0 = 1.0 / (s[s < np.log10(0.5)] - s.min()).mean()
    l1 = 1.0 / (s[s > np.log10(1.0)] - np.log10(1.0)).mean()
    alpha = np.exp(-l1)
    l = np.linspace(s.min(), s.max(), 101)
    #jp.ax_marg_x.plot(l, alpha * l1 * np.exp(-l1 * l))
    jp.ax_marg_x.plot(l, l0 * np.exp(-l0 * (l - s.min())))
    #jp.ax_marg_x.plot(l, (1 - alpha) * l0 * np.exp(-l0 * l) + alpha * l1 * np.exp(-l1 * l))
    #jp.ax_marg_x.set_ylim(0, alpha * l1)
    print(l0, l1, alpha)
    s0 = np.log((1-alpha)*l0/alpha/l1) / (l0 - l1)
    print(s0)

    #d = np.log10(d[s > s0][1:])
    jp.ax_marg_y.hist(d, density=True, bins=100, orientation='horizontal')
    l0 = 1.0 / (d[d < np.log10(1.0)] - d.min()).mean()
    l1 = 1.0 / (d[d > np.log10(1.0)] - np.log10(1.0)).mean()
    alpha = 1.0 / l1
    l = np.linspace(d.min(), d.max(), 101)
    #jp.ax_marg_y.plot(alpha * l1 * np.exp(-l1 * l), l)
    jp.ax_marg_y.plot(l0 * np.exp(-l0 * (l - d.min())), l)
    #jp.ax_marg_y.plot((1 - alpha) * l0 * np.exp(-l0 * (l - d.min())) + alpha * l1 * np.exp(-l1 * l), l)
    #jp.ax_marg_y.set_xlim(0, alpha * l1 * np.exp(-l1 * d.min()))
    print(l0, l1, alpha)
    d0 = (l0 * d.min() + np.log((1-alpha)*l0/alpha/l1)) / (l0 - l1)
    print(d0)

    jp.plot_joint(plt.scatter, c=np.log(r[1:]), edgecolor='k', cmap='bwr', alpha=0.5)
    #jp.ax_joint.axvline(s[0])
    jp.ax_joint.axhline(np.log10(0.8))
    jp.ax_joint.axvline(np.log10(0.3))
    jp.ax_joint.set_xticks(np.log10([0.25, 0.5, 1.0, 2.0, 4.0]))
    jp.ax_joint.set_xticklabels([0.25, 0.5, 1.0, 2.0, 4.0])
    jp.ax_joint.set_yticks(np.log10([0.5, 1.0, 2.0, 4.0, 8.0, 16.0]))
    jp.ax_joint.set_yticklabels([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    buf = io.BytesIO()
    jp.savefig(buf, format='png')
    #plt.savefig(buf, format='png')
    buf.seek(0)
    return tf.image.decode_png(buf.getvalue(), channels=4)


class PeakCommand(Command):

    name = 'peak'
    description = 'Find peaks'
    help = '''
'''

    options = [
        _option('job-dir', 'j', ''),
        option('force', 'f', ''),
    ]

    def is_error(self, stage):
        return stage < 1

    def is_target(self, stage):
        return stage == 1

    def force_stage(self, stage):
        return 1

    def create(self, data, prev, curr, logs,
               window, shift, gauss, radius, min_intensity, min_distance):
        name = self.status.params['name']
        batch = self.status.params['batch']
        verbose = self.status.params['pbar']
        imgs, mask, nt = moving_average_imgs(data, window, shift)
        pos, score = find_peak(
            imgs, mask, gauss, radius, min_intensity, batch, nt, verbose,
        )
        idx = reduce_peak_idx(pos, radius, min_distance)
        pos = pos[idx]
        score = score[idx]
        dist = calc_distance(pos, radius)
        radius = np.array(radius)[pos[:, 1]]
        save_pickle(f'{curr}-filter', (window, shift, gauss, radius, min_intensity, min_distance))
        save_numpy(f'{curr}-peak', pos[:, [0, 2, 3]])
        save_numpy(f'{curr}-stat', np.stack([score, dist, radius], axis=1))

        dt = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        logs = os.path.join(self.logs_dir, name, f'peak-{dt}')
        writer = tf.summary.create_file_writer(logs)
        with writer.as_default():
            img = draw(score, dist, radius)
            tf.summary.image(f'peak/stat', img[None, ...], step=0)
            writer.flush()
        writer.close()

