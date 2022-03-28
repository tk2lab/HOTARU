from .base import FigCommandBase


class FigStatCommand(FigCommandBase):

    name = 'figstat'
    _type = 'data'
    _suff = '_stat'
    description = 'Plot fig'
    help = '''
'''

    def _handle(self, base, p, fig):
        mask = p['mask']
        mmax = p['mmax']
        mstd = p['mstd']
        mcor = p['mcor']
        h, w = mask.shape
        hi, wi = h / 100, w / 100

        for i, img in enumerate([mmax, mstd, mcor]):
            ax = fig.add_axes([0, i * (hi + 0.5), wi, hi])
            im = ax.imshow(img)
            fig.colorbar(im)
