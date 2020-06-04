import numpy as np

from .base import Command, option
from ..train.model import HotaruModel


class FootprintCommand(Command):

    description = 'Update footprint'

    name = 'footprint'
    options = [
        option('job-dir'),
        option('name', flag=False, default='default'),
        option('batch', flag=False, default=100),
    ]

    def handle(self):
        self.set_job_dir()
        la = self.status['root']['la']
        lu = self.status['root']['lu']
        bx = self.status['root']['bx']
        bt = self.status['root']['bt']
        prev_key = self.status['spike_current']
        self.key = prev_key + ('footprint', (la, lu, bx, bt))
        self._handle('footprint')

    def create(self):
        print('footprint')
        spike = self.spike
        nk = spike.shape[0]
        nx = self.status['root']['nx']
        nt = self.status['root']['nt']
        tau1, tau2, hz, tauscale = self.key[-3][:4]
        la, lu, bx, bt = self.key[-1]
        batch = int(self.option('batch'))
        self.model = HotaruModel(
            self.data_file, nk, nx, nt,
            tau1, tau2, hz, tauscale, la, lu, bx, bt, batch,
        )
        self.model.compile()
        self.model.spike.val = spike
        self.model.update_footprint()
        return self.model.footprint.val

    def save(self, base, val):
        np.save(base + '.npy', val)
