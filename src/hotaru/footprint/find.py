from logging import getLogger
from math import inf

from keras import ops
from keras.callbacks import History
from keras.initializers import Constant

from ..data import MovieDataset
from ..data import MovieWithStats
from ..model import Model
from ..ops import gaussian_laplace_2d
from ..ops import max_pool_3d
from ..saving import Config
from ..saving import Data
from ..typing import Array
from ..typing import Tensor
from .radius import Radius
from .reduce import PeakList
from .reduce import reduce_peak

logger = getLogger(__name__)


class PeakMap(Data):
    radius: Radius
    ts: Array
    rs: Array
    gs: Array

    def reduce(
        self,
        min_radius: float,
        max_radius: float,
        min_distance_ratio: float,
        block_size: int,
    ) -> tuple[PeakList, PeakList]:
        return reduce_peak(self, min_radius, max_radius, min_distance_ratio, block_size)


class PeakFinder(Model):
    def __init__(self, radius: Radius | Config, **kwargs):
        super().__init__(**kwargs)
        self.radius = Radius.get(radius)

    def fit(self, data: MovieWithStats | Config, batch_size: int, **kwargs) -> History:
        if self.built:
            raise RuntimeError()
        sdata = MovieWithStats.get(data)
        self.build(sdata.shape)
        self.mask.assign(sdata.mask)
        dataset = MovieDataset(sdata, batch_size)
        return super().fit(dataset, **kwargs)

    def get_peakmap(self) -> PeakMap:
        ts, rs, vs = (ops.convert_to_numpy(v) for v in (self.ts, self.rs, self.vs))
        return PeakMap(self.radius, ts, rs, vs)

    def build(self, input_shapes) -> None:
        *_, h, w = input_shapes
        sentinel = Constant(-1)
        ninf = Constant(-inf)
        self.mask = self.add_weight(shape=(h, w), dtype='uint8', initializer='ones')
        self.ts = self.add_weight(shape=(h, w), dtype='int32', initializer=sentinel)
        self.rs = self.add_weight(shape=(h, w), dtype='int32', initializer=sentinel)
        self.gs = self.add_weight(shape=(h, w), dtype='float32', initializer=ninf)

    def custom_train_step(self, ts, imgs) -> dict:
        i, r, g = self(imgs)
        cond = g < self.gs
        self.ts.assign(ops.where(cond, self.ts, ts[i]))
        self.rs.assign(ops.where(cond, self.rs, r))
        self.gs.assign(ops.where(cond, self.gs, g))
        return {}

    def call(self, imgs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        radius = ops.convert_to_tensor(self.radius)
        nt, h, w = imgs.shape
        nr = radius.size
        gl = gaussian_laplace_2d(imgs, radius, axis=1)
        gl_max = max_pool_3d(gl, 3)
        gl_peak = gl == gl_max
        gl_peak &= self.mask.value
        gl = ops.where(gl_peak, gl, -inf)
        gl_reshape = gl.reshape(nt * nr, h, w)
        idx = ops.argmax(gl_reshape, axis=0)
        gl_max = ops.take_along_axis(gl_reshape, idx[None, ...], axis=0)[0]
        t, r = idx // nr, ops.mod(idx, nr)
        return t, r, gl_max
