from cleo import Command
from cleo import option

from .options import options
from .options import tag_options


class RunCommand(Command):

    name = 'run'
    description = 'Run'
    help = '''
'''

    options = [
        options['tag'],
        tag_options['start'],
        tag_options['goal'],
    ]

    def handle(self):
        tag = self.option('tag')
        start = int(self.option('start'))
        goal = int(self.option('goal'))
        if start == 0:
            self.call('data')
            self.call('find')
            self.call('reduce')
            self.call('init', f'--tag {tag}000')
            self.call('temporal', f'--tag {tag}000 --footprint-tag {tag}000')
            start = 1
        for s in range(start, goal + 1):
            prev = f'{tag}{s-1:03}'
            curr = f'{tag}{s:03}'
            self.call('spatial', f'--tag {curr} --spike-tag {prev}')
            self.call('clean', f'--tag {curr} --footprint-tag {curr}')
            self.call('temporal', f'--tag {curr} --footprint-tag {curr}')
