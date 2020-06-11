import os

import tensorflow as tf
import pandas as pd

from .base import Command, option
from ..data.dataset import unmasked
from ..peak.find import find_peak
from ..peak.reduce import reduce_peak


class PeakCommand(Command):

    description = 'Find peaks'

    name = 'peak'
    options = [
        option('job-dir', flag=False, value_required=False),
        option('thr-gl', flag=False, default=0.4),
        option('thr-dist', flag=False, default=1.6),
        option('name', flag=False, default='default'),
        option('batch', flag=False, default=100),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        gauss = float(self.status['root']['gauss'])
        radius = tuple(float(x) for x in self.status['root']['diameter'])
        thr_gl = float(self.option('thr-gl'))
        thr_dist = float(self.option('thr-dist'))
        self.key = 'peak', (gauss, radius, thr_gl, thr_dist)
        self._handle('peak')

    def create(self):
        self.line('peak')
        name = self.option('name')
        base = os.path.join(self.work_dir, 'peak', name)
        gauss, radius, thr_gl, thr_dist = self.key[1]
        peak = self.handle0(gauss, radius, thr_gl)
        peak = reduce_peak(peak, thr_dist)
        return peak

    def handle0(self, gauss, radius, thr_gl):
        status = self.status['peak']
        key = 'peak', (gauss, radius, thr_gl)
        name = self.option('name') + '0'
        base = os.path.join(self.work_dir, 'peak', name)
        if self.option('force') or key not in status:
            batch = int(self.option('batch'))
            nt = self.status['root']['nt']
            prog = tf.keras.utils.Progbar(nt)
            peak = find_peak(self.data, self.mask, gauss, radius, thr_gl, batch, prog)
            self.save(base, peak)
            status[key] = name
            self.save_status()
        else:
            peak = self.load_peak(base)
        return peak

    def save(self, base, peak):
        _peak_name = 'ts', 'rs', 'xs', 'ys', 'gs'
        peak_dict = {k: v for k, v in zip(_peak_name, peak)}
        with tf.io.gfile.GFile(base + '.csv', 'w') as fp:
            pd.DataFrame(peak_dict).to_csv(fp)
