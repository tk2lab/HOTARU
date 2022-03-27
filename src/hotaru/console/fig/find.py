from .base import FigCommandBase
from ...util.csv import load_csv
from ...eval.radius import plot_radius


class FigFindCommand(FigCommandBase):

    name = 'figfind'
    _type = 'peak'
    description = 'Plot fig'
    help = '''
'''

    def _handle(self, base, p, fig):
        radius = p['radius']

        peaks = load_csv(f'{base}.csv')
        peaks['dummy'] = 'dummy'

        ax = fig.add_axes([0.5, 0.5, 5, 3])
        plot_radius(
            ax, peaks, 'intensity', 'dummy', radius,
            edgecolor='none', alpha=0.5, size=2,
            legend=False, rasterized=True
        )
