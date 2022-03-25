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
        options['data_tag'],
        options['peak_tag'],
        options['init_tag'],
        tag_options['start'],
        tag_options['goal'],
    ]

    def handle(self):
        data_tag = self.option('data-tag')
        peak_tag = self.option('peak-tag')
        init_tag = self.option('init-tag')
        tag = self.option('tag')
        start = int(self.option('start'))
        goal = int(self.option('goal'))
        if start == 0:
            self.call('data', f'--tag {data_tag}')
            self.call('find', f'--tag {peak_tag} --data-tag {data_tag}')
            self.call('reduce', f'--tag {init_tag} --peak-tag {peak_tag}')
            self.call('init', f'--tag {init_tag}000 --data-tag {data_tag} --peak-tag {init_tag}')
            self.call('temporal', f'--tag {tag}000 --data-tag {data_tag} --footprint-tag {init_tag}000')
            start = 1
        for s in range(start, goal + 1):
            prev = f'{tag}{s-1:03}'
            curr = f'{tag}{s:03}'
            self.call('spatial', f'--tag {curr} --data-tag {data_tag} --spike-tag {prev}')
            self.call('clean', f'--tag {curr} --footprint-tag {curr}')
            self.call('temporal', f'--tag {curr} --data-tag {data_tag} --footprint-tag {curr}')
