import pickle

from .base import Command, option


class StatusCommand(Command):

    description = 'Show Status'

    name = 'status'
    options = [
        option('job-dir', flag=False, value_required=False),
    ]

    def handle(self):
        self.set_job_dir()
        status = self.status

        key = status['root']
        for k, v in self.status['root'].items():
            self.line('{}: {}'.format(k, v))
        key = status['peak_current']
        if key is not None:
            self.line('peak-current: {}'.format(self.status['peak'][key]))
            for k, v in self.status['peak'].items():
                self.line('peak/{}:'.format(v))
                for n, h in zip(k[::2], k[1::2]):
                    self.line('  {}: {}'.format(n, h))
        key = status['footprint_current']
        if key is not None:
            self.line('footprint_current: {}'.format(status['footprint'][key]))
            for k, v in status['footprint'].items():
                self.line('footprint/{}:'.format(v))
                for n, h in zip(k[::2], k[1::2]):
                    self.line('  {}: {}'.format(n, h))
        key = status['spike_current']
        if key is not None:
            self.line('spike_current: {}'.format(status['spike'][key]))
            for k, v in status['spike'].items():
                self.line('spike/{}:'.format(v))
                for n, h in zip(k[::2], k[1::2]):
                    self.line('  {}: {}'.format(n, h))
