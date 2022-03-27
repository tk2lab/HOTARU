import numpy as np

from .base import FigCommandBase
from ...util.csv import load_csv
from ...util.numpy import load_numpy
from ...eval.radius import plot_radius
from ...eval.circle import plot_circle
from ...eval.footprint import plot_maximum
#from ...eval.footprint import plot_contour


class FigInitCommand(FigCommandBase):

    name = 'figinit'
    _type = 'footprint'
    description = 'Plot fig'
    help = '''
'''

    def _handle(self, base, p, fig):
        mask = p['mask']
        radius = p['radius']
        distance = p['distance']
        h, w = mask.shape
        aspect = h / w

        peaks = load_csv(f'{base}_peaks.csv')
        val = load_numpy(f'{base}.npy')
        nk = val.shape[0]
        footprint = np.zeros((nk, h, w), np.float32)
        footprint[:, mask] = val

        ax = fig.add_axes([0.5, 0.5, 5, 3])
        plot_radius(
            ax, peaks, 'intensity', 'accept', radius,
            edgecolor='none', alpha=0.5, size=2,
            legend=False, rasterized=True,
        )

        ax = fig.add_axes([0.5, 4.0, 5, 5 * aspect])
        plot_circle(ax, peaks.query('accept=="yes"'), h, w, distance)

        ax = fig.add_axes([0.5, 4.5 + 5 * aspect, 5, 5 * aspect])
        plot_maximum(ax, footprint)

        '''
        ax = fig.add_axes([1.0 + 5 * aspect, 4.0, 5, 5 * aspect])
        plot_contour(ax, footprint)
        '''
