from .base import Command, _option


class RunCommand(Command):

    description = 'Execute all'

    name = 'run'
    options = [
        _option('job-dir'),
        _option('name'),
        _option('rep-num'),
    ]

    def handle(self):
        self.set_job_dir()
        if self.status['clean_current'] is None:
            self.call('data')
            self.call('peak')
            self.call('segment')
        self.call('spike')
        nk = self.spike.shape[0]
        for i in range(int(self.option('rep-num') or 5)):
            self.call('footprint')
            self.call('clean')
            self.call('spike')
            nk, old_nk = self.spike.shape[0], nk
            self.line(f'<comment>number of cell: {old_nk} -> {nk}</comment>')
            if nk == old_nk:
                break
        self.call('output')
