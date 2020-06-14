from .base import Command, option


class RunCommand(Command):

    description = 'Execute all'

    name = 'run'
    options = [
        option('job-dir', flag=False, value_required=False),
        option('name', flag=False, value_required=False),
    ]

    def handle(self):
        self.set_job_dir()
        if self.status['clean_current'] is None:
            self.call('data')
            self.call('peak')
            self.call('segment')
        self.call('spike')
        nk = self.spike.shape[0]
        while True:
            self.call('footprint')
            self.call('clean')
            self.call('spike')
            nk, old_nk = self.spike.shape[0], nk
            self.line(f'<comment>number of cell: {old_nk} -> {nk}</comment>')
            if nk == old_nk:
                break
