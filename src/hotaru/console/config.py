from cleo import Command

from .options import tag_options
from .options import options
from ..util.pickle import save_pickle


class ConfigCommand(Command):

    name = 'config'
    description = 'Set default parameters'
    help = '''
'''

    options = list(options.values())

    def handle(self):
        p = {k: self.option(k.replace('_', '-')) for k in options.keys()}
        for k, v in p.items():
            self.line(f'{k:15}: {v}')
        save_pickle('hotaru/config.pickle', p)
