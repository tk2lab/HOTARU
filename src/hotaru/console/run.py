from .base import Command, _option


class RunCommand(Command):

    description = 'Execute all'

    name = 'run'
    options = [
        _option('job-dir'),
        _option('goal'),
    ]

    def handle(self):
        self.set_job_dir()

        name = self.status.params['name']
        stage = len(self.status.history.get(name, ()))
        goal_stage = int(self.option('goal') or 60)

        for s in range(stage, goal_stage):
            if s == 0:
                self.call('data')
            elif s == 1:
                self.call('peak')
            elif s == 2:
                self.call('segment')
            elif s % 3 == 0:
                self.call('spike')
            elif s % 3 == 1:
                self.call('footprint')
            elif s % 3 == 2:
                self.call('clean')

        self.call('output')
        self.print_gpu_memory()
