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

    options = [options['tag'], options['stage']]

    def handle(self):
        tag = self.option('tag')
        stage = int(self.option('stage'))
        curr = '' if stage < 0 else f'_{stage:03}'

        base = f'hotaru/{self._type}/{tag}{curr}{self._suff}'
        log = load_pickle(f'{base}_log.pickle')
        fig = plt.figure(figsize=(1,1))
        self._handle(base, log, fig)
        os.makedirs('hotaru/fig', exist_ok=True)
        fig.savefig(f'hotaru/fig/{tag}{curr}_{self.name[3:]}.pdf', bbox_inches='tight')
