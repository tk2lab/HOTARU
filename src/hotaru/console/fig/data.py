from .base import FigCommandBase


class FigDataCommand(FigCommandBase):

    name = 'figdata'
    _type = 'data'
    description = 'Plot fig'
    help = '''
'''

    def _handle(self, base, p, fig):
        mask = p['mask']
        avgt = p['avgt']
        avgx = p['avgx']
        sstd = p['sstd']
        mmax = p['mmax']
        mcor = p['mcor']
        h, w = mask.shape
        aspect = h / w

        ax = fig.add_axes([0.5, 0.5, 5, 2])
        ax.plot(avgt / sstd)

        ax = fig.add_axes([0.5, 3.0, 5, 5 * aspect])
        im = ax.imshow(avgx / sstd)
        fig.colorbar(im)

        ax = fig.add_axes([0.5, 3.5 + 5 * aspect, 5, 5 * aspect])
        im = ax.imshow((mmax - avgx) / sstd)
        fig.colorbar(im)

        ax = fig.add_axes([0.5, 4.0 + 10 * aspect, 5, 5 * aspect])
        im = ax.imshow(mcor)
        fig.colorbar(im)
