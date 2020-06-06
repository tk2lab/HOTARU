import numpy as np

from .base import Command, option


class SpikeCommand(Command):

    description = 'Update spike'

    name = 'spike'
    options = [
        option('job-dir'),
        option('name', flag=False, default='default'),
        option('batch', flag=False, default=100),
        option('force', 'f', 'overwrite previous result'),
    ]

    def handle(self):
        self.set_job_dir()
        tau1 = self.status['root']['tau-fall']
        tau2 = self.status['root']['tau-rise']
        hz = self.status['root']['hz']
        tauscale = self.status['root']['tau-scale']
        la = self.status['root']['la']
        lu = self.status['root']['lu']
        bx = self.status['root']['bx']
        bt = self.status['root']['bt']
        prev_key = self.status['footprint_current']
        self.key = prev_key + ('spike', (tau1, tau2, hz, tauscale, la, lu, bx, bt))
        self._handle('spike')

    def create(self):
        self.line('spike')
        model = self.model
        model.footprint.val = self.footprint
        model.update_spike()
        return model.spike.val

    def save(self, base, val):
        np.save(base + '.npy', val)
