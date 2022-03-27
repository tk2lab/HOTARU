from .base import FigCommandBase
from ...util.csv import load_csv
from ...eval.radius import plot_radius
from ...eval.circle import plot_circle


class FigTemporalCommand(FigCommandBase):

    name = 'figtemporal'
    _type = 'spike'
    description = 'Plot fig'
    help = '''
'''

    def _handle(self, base, p, fig):
        '''
        mask = p['mask']
        radius = p['radius']
        distance = p['distance']
        h, w = mask.shape
        aspect = h / w

        peaks = load_csv(f'{base}.csv')

        ax = fig.add_axes([0.5, 0.5, 5, 3])
        plot_radius(
            ax, peaks, 'intensity', 'accept', radius,
            edgecolor='none', alpha=0.5, size=2,
            legend=False, rasterized=True,
        )

        ax = fig.add_axes([0.5, 4.0, 5, 5 * aspect])
        plot_circle(ax, peaks.query('accept=="yes"'), h, w, distance)
        '''
        pass
