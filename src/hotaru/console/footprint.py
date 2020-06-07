import numpy as np

from .base import Command, option


class FootprintCommand(Command):

    description = 'Update footprint'

    name = 'footprint'
    options = [
        option('job-dir'),
        option('name', flag=False, default='default'),
        option('batch', flag=False, default=100),
        option('force', 'f', 'overwrite previous result'),
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
        model = self.model
        model.spike.val = self.spike
        model.update_footprint(batch=int(self.option('batch')))
        return model.footprint.val

    def save(self, base, val):
        np.save(base + '.npy', val)
