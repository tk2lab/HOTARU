from .base import Command, option


class RunCommand(Command):

    description = 'Execute all'

    name = 'run'
    options = [
        option(n, flag=False, value_required=False)
        for n in [
            'job-dir', 'imgs-file', 'hz', 'mask-type',
            'gauss', 'radius', 'thr-gl', 'thr-dist',
            'tau1', 'tau2', 'tauscale', 'la', 'lu', 'bx', 'bt',
            'epochs', 'batch', 'start',
        ]
    ]

    def handle(self):
        self.set_job_dir()
        self._call('parameter', 'gauss', 'radius', 'tau1', 'tau2', 'hz', 'tauscale', 'la', 'lu', 'bx', 'bt')
        start = self.option('start')
        if start:
            footprint_key = {v: k for k, v in self.status['footprint'].items()}[start]
            spike_key = {v: k for k, v in self.status['spike'].items()}[start]
            self.status['footprint_current'] = footprint_key
            self.status['spike_current'] = spike_key
            self.save_status()
        elif self.status['footprint_current'] is None:
            self._call('data', 'imgs-file', 'mask-type', 'batch')
            self._call('peak', 'thr-gl', 'thr-dist', 'batch')
            self._call('make', 'batch')
        for i in range(int(self.option('epochs') or 3)):
            self._call('spike', 'batch')
            self._call('footprint', 'batch')
            self._call('clean', 'batch')
        self._call('spike', 'batch')

    def _call(self, command, *args):
        args = ' '.join('--{}={}'.format(a, self.option(a)) for a in args if self.option(a))
        self.call(command, args)
