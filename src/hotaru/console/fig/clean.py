import numpy as np

from .base import FigCommandBase
from ...util.numpy import load_numpy
from ...util.csv import load_csv
from ...eval.radius import plot_radius
from ...eval.footprint import plot_maximum
#from ...eval.footprint import plot_contour


class FigCleanCommand(FigCommandBase):

    name = 'figclean'
    _type = 'footprint'
    description = 'Plot fig'
    help = '''
'''

    def _handle(self, base, p, fig):
        mask = p['mask']
        h, w = mask.shape
        aspect = h / w
        radius = p['radius']

        val = load_numpy(f'{base}.npy')
        nk = val.shape[0]
        footprint = np.zeros((nk, h, w), np.float32)
        footprint[:, mask] = val

        peaks = load_csv(f'{base}_peaks.csv')

        ax = fig.add_axes([0.5, 0.5, 5, 3])
        plot_radius(
            ax, peaks, 'firmness', 'accept', radius,
            edgecolor='none', alpha=0.5, size=2, legend=False,
        )

        ax = fig.add_axes([0.5, 4.0, 5, 5 * aspect])
        plot_maximum(ax, footprint)

        '''
        ax = fig.add_axes([1.0 + 5 * aspect, 4.0, 5, 5 * aspect])
        plot_contour(ax, footprint)
        '''
