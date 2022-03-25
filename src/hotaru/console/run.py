from cleo import Command

from .options import options


class RunCommand(Command):

    name = 'run'
    description = 'Run'
    help = '''
'''

    options = [
        options['tag'],
    ]

    def handle(self):
        tag = self.option('tag')
        stage = 10
        self.call('data')
        self.call('find')
        self.call('reduce')
        self.call('init', f'--tag {tag}000')
        self.call('temporal', f'--tag {tag}000 --footprint-tag {tag}000')
        for s in range(1, stage):
            prev = f'{tag}{s-1:03}'
            curr = f'{tag}{s:03}'
            self.call('spatial', f'--tag {curr} --spike-tag {prev}')
            self.call('clean', f'-f --tag {curr} --footprint-tag {curr}')
            self.call('temporal', f'--tag {curr} --footprint-tag {curr}')
