import pickle

from .base import Command, _option


class HistoryCommand(Command):

    description = 'Show History'

    name = 'history'
    options = [
        _option('job-dir'),
    ]

    def handle(self):
        self.set_job_dir()
        status = self.status
        for typ in ['peak', 'clean', 'spike', 'footprint']:
            key = status[f'{typ}_current']
            if key is not None:
                self.line(f'{typ}_current: {status[typ][key]}')
                for k, v in status[typ].items():
                    self.line(f'{typ}/{v}:')
                    for n in k:
                        self.line('  {}: {}'.format(n[0], n[1:]))
