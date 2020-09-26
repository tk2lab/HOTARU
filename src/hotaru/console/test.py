import datetime
import os
import io

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from .base import Command, option, _option
from ..footprint.reduce import reduce_peak_idx
from ..util.numpy import load_numpy
from ..util.pickle import load_pickle


def draw_img(w, h, x, y, r, s, thr_distance):
    fig = plt.figure(figsize=(w/72, h/72), dpi=72)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        x, y, s=2 * np.pi * (thr_distance * r)**2, c=s / s.max(),
        cmap='Greens', edgecolors='k', alpha=0.5,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return tf.image.decode_png(buf.getvalue(), channels=4)


class TestCommand(Command):

    name = 'test'
    description = 'Test to make segment'
    help = '''
'''

    options = [
        _option('job-dir', 'j', ''),
    ]

    def handle(self):
        self.set_job_dir()

        name = self.status.params['name']
        max_intensity, max_distance = self.status.get_test_params()

        stage = len(self.status.history.get(name, ()))
        if stage <= 0:
            self.call('data')
        if stage <= 1:
            self.call('peak')

        self.line('test', 'info')

        history = self.status.history.get(name, ())
        data = self.status.find_saved(history[:1])
        data = os.path.join(self.work_dir, data, f'000')
        h, w = load_numpy(f'{data}-mask').shape
        prev = self.status.find_saved(history[:2])
        prev = os.path.join(self.work_dir, prev, f'001')
        params = load_pickle(f'{prev}-filter')
        radius, min_intensity, min_distance = params[3:6]
        radius = np.array(radius)
        nr = radius.size
        p0 = load_numpy(f'{prev}-peak')
        self.line(f'{p0.shape}')
        r0 = radius[p0[:, 1]]
        s0 = load_numpy(f'{prev}-intensity')

        dt = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        logs = os.path.join(self.logs_dir, name, f'test-{dt}')
        writer = tf.summary.create_file_writer(logs)
        distances = np.linspace(min_distance, max_distance, 6)
        intensities = np.linspace(max_intensity, min_intensity, 10)
        with writer.as_default():
            for step1, thr_distance in enumerate(distances):
                idx = reduce_peak_idx(p0, radius, thr_distance)
                p1 = p0[idx]
                self.line(f'{thr_distance} {p1.shape}')
                r1 = r0[idx]
                s1 = s0[idx]
                tf.summary.histogram(f'test/radius', r1, step=step1)
                tf.summary.histogram(f'test/intensity', s1, step=step1)
                writer.flush()
                for step2, thr_intensity in enumerate(intensities):
                    cond = s1 > thr_intensity
                    cond &= (0 < p1[:, 1]) & (p1[:, 1] < nr - 1)
                    y, x = p1[cond, 2], p1[cond, 3]
                    r = r1[cond]
                    s = s1[cond]
                    img = draw(w, h, x, y, r, s, thr_distance)[None, ...]
                    tf.summary.image(
                        f'test/{thr_distance:.3f}', img, step=step2,
                    )
                    writer.flush()
        writer.close()
