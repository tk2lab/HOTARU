from logging import getLogger
from math import nan

from keras import ops
from matplotlib.pyplot import get_cmap

from ..saving import PathLike
from ..typing import Tensor
from .calc_stats import StatsCalculator
from .data import CalciumImagingData
from .io import to_movie

logger = getLogger(__name__)


class CalciumImagingDataWithStats(CalciumImagingData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        nt, h, w = self.shape
        self.avgt = self.add_weight((nt,), name='avgt')
        for key in ('avgx', 'imin', 'imax', 'istd', 'icor'):
            setattr(self, key, self.add_weight((h, w), name=key))
        for key in ('std0', 'min0', 'max0'):
            setattr(self, key, self.add_weight((), name=key))
        super()._build_at_init()

    def calc(self, **kwargs) -> None:
        kwargs.setdefault('desc', f'Stats ({self.imgs_path})')
        calc = StatsCalculator(self.imgs, self.mask)
        calc.compile(**kwargs.pop('compile_kwargs', {}))
        calc.fit(**kwargs)
        keys = str.split('avgt, avgx, std0, min0, max0, imin, imax, istd, icor', ', ')
        vals = calc.post_fit()
        for key, val in zip(keys, vals, strict=True):
            getattr(self, key).assign(val)

    def normalize(self, ts: Tensor, imgs: Tensor) -> Tensor:
        imgs = (ops.cast(imgs, 'float32') - self.avgt[ts, None, None] - self.avgx) / self.std0
        imgs = ops.where(ts[:, None, None] >= 0, imgs, nan)
        return imgs

    def to_movie(self, path: PathLike, cmap='Greens', *, normalize: bool = True, **kwargs) -> None:
        imgs = self.imgs
        nt = self.shape[0]
        cmap = get_cmap(cmap)

        if normalize:
            avgt = self.avgt.numpy()
            avgx = self.avgx.numpy()
            std0 = self.std0.numpy()
            min0 = self.imin.numpy().min()
            scale = self.imax.numpy().max() - min0

            def frame(t):
                val = (imgs[t] - avgt[t] - avgx) / std0
                val = (val - min0) / scale
                return val
        else:
            vmin = self.min0.numpy()
            scale = self.max0.numpy() - self.min0.numpy()

            def frame(t):
                val = (imgs[t] - vmin) / scale
                return val

        def gen():
            for t in range(nt):
                yield (255 * cmap(frame(t))).astype('uint8')

        to_movie(path, gen(), imgs.shape, self.hz, **kwargs)
