import pickle

from .base import Command, _option


class StatusCommand(Command):

    description = 'Show Status'

    name = 'status'
    options = [
        _option('job-dir'),
    ]

    def handle(self):
        self.set_job_dir()
        self.call('config')
        status = self.status
        key = status['root']
        for k, v in self.status['root'].items():
            self.line('{}: {}'.format(k, v))
