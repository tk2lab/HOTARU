import os

from cleo import Command

from .options import tag_options
from .options import options
from .options import option_type
from ..util.pickle import save_pickle


class ConfigCommand(Command):

    name = 'config'
    description = 'Set default parameters'
    help = '''
'''

    options = list(options.values())

    def handle(self):
        p = {
            k: f(self.option(k.replace('_', '-')))
            for k, f in option_type.items()
        }
        for k, v in p.items():
            self.line(f'{k:15}: {v}')
        os.makedirs('hotaru', exist_ok=True)
        save_pickle('hotaru/config.pickle', p)
