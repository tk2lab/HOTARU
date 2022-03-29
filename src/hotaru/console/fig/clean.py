import numpy as np

from hotaru.util.numpy import load_numpy
from hotaru.util.csv import load_csv
from hotaru.eval.radius import plot_radius
from hotaru.eval.footprint import plot_maximum
from hotaru.eval.footprint import plot_contour
from hotaru.eval.footprint import calc_sim_cos

from .base import FigCommandBase


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
        np.set_printoptions(3, suppress=True)
        print(np.sort(calc_sim_cos(val)))
        nk = val.shape[0]
        footprint = np.zeros((nk, h, w), np.float32)
        footprint[:, mask] = val

        peaks = load_csv(f'{base}_peaks.csv')

        ax = fig.add_axes([0.5, 0.5, 5, 3])
        plot_radius(
            ax, peaks, 'firmness', 'accept', radius,
            edgecolor='none', alpha=0.5, size=2, legend=False,
        )

        thr = 0.7
        footprint -= thr
        footprint[footprint < 0.0] = 0.0
        footprint /= (1 - thr)
        ax = fig.add_axes([0.5, 4.0, 5, 5 * aspect])
        plot_maximum(ax, footprint)

        ax = fig.add_axes([0.5, 4.5 + 5 * aspect, 5, 5 * aspect])
        plot_contour(ax, footprint, gauss=2.0, thr_out=0.9)
