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
        self.call('config')
        status = self.status
        key = status['root']
        for k, v in self.status['root'].items():
            self.line('{}: {}'.format(k, v))
