import os

from cleo import Command
import matplotlib.pyplot as plt

from ..options import options
from ...util.pickle import load_pickle


class FigCommandBase(Command):

    _suff = ''
    description = 'Plot fig'
    help = '''
'''

    options = [options['tag']]

    def handle(self):
        tag = self.option('tag')

        base = f'hotaru/{self._type}/{tag}{self._suff}'
        log = load_pickle(f'{base}_log.pickle')
        fig = plt.figure(figsize=(1,1))
        self._handle(base, log, fig)
        os.makedirs('hotaru/fig', exist_ok=True)
        fig.savefig(f'hotaru/fig/{tag}_{self.name[3:]}.pdf', bbox_inches='tight')
