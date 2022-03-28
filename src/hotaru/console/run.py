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
        options['data-tag'],
        options['peak-tag'],
        options['init-tag'],
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
        for s in range(start, goal + 1):
            self.line(f'{tag} {s:03}')
            if s == 0:
                self.call('data', f'-t {data_tag}')
                self.call('find', f'-t {peak_tag} -D {data_tag}')
                self.call('init', f'-t {init_tag} -D {data_tag} -P {peak_tag}')
                self.call('temporal', f'-t {tag} -s 0 -D {data_tag} -P {init_tag}')
            else:
                self.call('spatial', f'-t {tag} -s {s} -D {data_tag} -P {tag}')
                self.call('clean', f'-t {tag} -s {s} -P {tag}')
                self.call('temporal', f'-t {tag} -s {s} -D {data_tag} -P {tag}')
