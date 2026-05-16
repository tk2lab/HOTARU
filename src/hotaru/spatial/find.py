from logging import getLogger
from math import inf
from math import nan

from keras import ops

from ..data import MovieDataset
from ..data import MovieWithStats
from ..data import Stats
from ..models import Model
from ..ops import gaussian_laplace_2d_multi
from ..ops import max_pool_3d
from ..saving import Config
from ..saving import cached_getter
from ..typing import Tensor
from .radius import Radius
from .reduce import PeakMap

logger = getLogger(__name__)


class PeakFinder(Model):
    @cached_getter(PeakMap)
    def get_peakmap(
        self,
        data: MovieWithStats | Config,
        radius: Radius | Config,
        batch_size: int,
        compile_kwargs: dict | None = None,
        dataset_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
    ) -> PeakMap:
        compile_kwargs = compile_kwargs or {}
        dataset_kwargs = dataset_kwargs or {}
        fit_kwargs = fit_kwargs or {}

        self.radius = Radius.get(radius)
        self.compile(**compile_kwargs)

        data = MovieWithStats.get(data)
        if not self.built:
            self.build(data.shape)
        self.reset(data.stats)

        dataset = MovieDataset(data, batch_size, **dataset_kwargs)
        fit_kwargs.setdefault('shuffle', False)
        self.fit(dataset, **fit_kwargs)

        tmap, rmap, gmap = (v.numpy() for v in (self.tmap, self.rmap, self.gmap))
        return PeakMap(self.radius, tmap, rmap, gmap)

    def build(self, input_shapes) -> None:
        _, h, w = input_shapes
        self.avgx = self.add_weight((h, w), dtype='float32', trainable=False, name='avgx')
        self.std0 = self.add_weight((), dtype='float32', trainable=False, name='std0')
        self.tmap = self.add_weight((h, w), dtype='int32', trainable=False, name='tmap')
        self.rmap = self.add_weight((h, w), dtype='int32', trainable=False, name='rmap')
        self.gmap = self.add_weight((h, w), dtype='float32', trainable=False, name='gmap')

    def reset(self, stats: Stats) -> None:
        self.avgx.assign(stats.avgx)
        self.std0.assign(stats.std0)
        self.gmap.assign(ops.full(self.gmap.shape, -inf, 'float32'))

    def custom_train_step(self, data) -> dict:
        ts, imgs = data
        imgs = (ops.cast(imgs, 'float32') - self.avgx) / self.std0
        imgs = ops.where(ts[:, None, None] >= 0, imgs, nan)
        i, r, g = self(imgs)
        t = ops.take(ts, i)
        cond = g > self.gmap.value
        self.tmap.assign(ops.where(cond, t, self.tmap.value))
        self.rmap.assign(ops.where(cond, r, self.rmap.value))
        self.gmap.assign(ops.where(cond, g, self.gmap.value))
        return {'loss': 0}

    def call(self, imgs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        imgs = ops.nan_to_num(imgs, nan=0)
        mask = ops.isfinite(self.avgx)
        radius = self.radius

        *shape, h, w = ops.shape(imgs)
        nr = radius.size

        gl = gaussian_laplace_2d_multi(imgs, radius, axis=1)
        gl_max = max_pool_3d(gl, 3)
        gl_peak = gl == gl_max
        gl_peak &= mask
        gl = ops.where(gl_peak, gl, -inf)
        gl_reshape = ops.reshape(gl, (-1, h, w))
        idx = ops.argmax(gl_reshape, axis=0)
        gl_max = ops.take_along_axis(gl_reshape, idx[None, ...], axis=0)[0]
        t, r = idx // nr, ops.mod(idx, nr)
        return t, r, gl_max
