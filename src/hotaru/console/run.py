from .base import Command, option


class RunCommand(Command):

    description = 'Execute all'

    name = 'run'
    options = [
        option('job-dir', flag=False, value_required=False),
        option('name', flag=False, value_required=False),
        option('start', flag=False, value_required=False),
        option('end', flag=False, value_required=False),
    ]

    def handle(self):
        self.set_job_dir()
        start = self.option('start')
        if start:
            clean_key = {v: k for k, v in self.status['clean'].items()}[start]
            self.status['clean_current'] = clean_key
            self.save_status()
        elif self.status['clean_current'] is None:
            self.call('data')
            self.call('peak')
            self.call('segment')
        for i in range(3):
            self.call('spike')
            self.call('footprint')
            self.call('clean')
        self.call('spike')
