import datetime
import os

import tensorflow as tf
import numpy as np
import matplotlib.cm as cm

from .base import Command, option, _option
from ..footprint.reduce import reduce_peak_idx
from ..util.numpy import load_numpy
from ..util.pickle import load_pickle


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

        stage = len(self.status.history.get(name, ()))
        if stage <= 0:
            self.call('data')
        if stage <= 1:
            self.call('peak')

        self.line('test', 'info')
        min_intensity, max_intensity, min_distance, max_distance = self.status.get_test_params()
        green = cm.Greens

        history = self.status.history.get(name, ())
        data = self.status.find_saved(history[:1])
        data = os.path.join(self.work_dir, data, f'000')
        h, w = load_numpy(f'{data}-mask').shape
        prev = self.status.find_saved(history[:2])
        prev = os.path.join(self.work_dir, prev, f'001')
        radius = np.array(load_pickle(f'{prev}-filter')[1])
        nr = radius.size
        p0 = load_numpy(f'{prev}-peak')
        r0 = radius[p0[:, 1]]
        s0 = load_numpy(f'{prev}-intensity')

        dt = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        logs = os.path.join(self.application.job_dir, 'logs', name, f'test-{dt}')
        writer = tf.summary.create_file_writer(logs)
        with writer.as_default():
            for step, thr_distance in enumerate(np.linspace(min_distance, max_distance, 12)):
                idx = reduce_peak_idx(p0, radius, thr_distance)
                p1 = p0[idx]
                r1 = r0[idx]
                s1 = s0[idx]
                tf.summary.histogram(f'test/radius', r1, step=step)
                tf.summary.histogram(f'test/intensity', s1, step=step)
                writer.flush()
                for thr_intensity in np.linspace(min_intensity, max_intensity, 10):
                    cond = s1 > thr_intensity
                    cond &= (0 < p1[:, 1]) & (p1[:, 1] < nr - 1)
                    img = np.zeros((h, w), np.int32)
                    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                    for (y, x), r in zip(p1[cond, 2:], r1[cond]):
                        cond = (xx - x) ** 2 + (yy - y) ** 2 < (1.5 * r) ** 2
                        img[cond] += 1
                    img = green(img / img.max())
                    tf.summary.image(f'test/{thr_intensity:.3f}-{thr_distance:.3f}', img[None, ...], step=0)
                    writer.flush()
        writer.close()
