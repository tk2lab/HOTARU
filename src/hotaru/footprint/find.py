from logging import getLogger
from math import inf

import numpy as np
from keras import ops
from keras.utils import PyDataset
from keras.utils import unpack_x_y_sample_weight

from ..io import MovieData
from ..model import Model
from ..ops import gaussian_laplace_2d
from ..ops import max_pool_3d
from ..saving import Config
from ..typing import Array
from ..typing import Tensor
from .radius import Radius
from .reduce import PeakMap

logger = getLogger(__name__)


class FinderDataset(PyDataset):
    def __init__(self, data: MovieData, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.batch_size = batch_size

    def on_epoch_end(self):
        pass

    def __len__(self) -> int:
        return self.data.num_frames // self.batch_size

    def __getitem__(self, index: int) -> tuple[Array, Array]:
        s = self.batch_size * index
        e = s + self.batch_size
        return np.arange(s, e, dtype='int32'), self.data.data[s:e]


class PeakFinder(Model):
    def __init__(self, radius: Radius | Config):
        self.radius = Radius.get(radius)

    def fit(self, data: MovieData | Config, batch_size: int, **kwargs):
        data = MovieData.get(data)
        dataset = FinderDataset(data, batch_size)
        super().fit(dataset, **kwargs)

    def get_peakval(self) -> PeakMap:
        ts, rs, vs = (ops.convert_to_numpy(v) for v in (self.ts, self.rs, self.vs))
        return PeakMap(self.radius, ts, rs, vs)

    def build(self, input_shapes) -> None:
        *_, h, w = input_shapes
        self.ts = self.add_weight(shape=(h, w), dtype='int32', initializer=-1)
        self.rs = self.add_weight(shape=(h, w), dtype='int32', initializer=-1)
        self.gs = self.add_weight(shape=(h, w), dtype='float32', initializer=-inf)

    def custom_train_step(self, data) -> dict:
        x, _y, _sample_weight = unpack_x_y_sample_weight(data)
        ts, imgs, mask = x
        i, r, g = self(imgs, mask)
        cond = g < self.gs
        self.ts.assign(ops.where(cond, self.ts, ts[i]))
        self.rs.assign(ops.where(cond, self.rs, r))
        self.gs.assign(ops.where(cond, self.gs, g))
        return {}

    def call(self, imgs: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        radius = ops.convert_to_tensor(self.radius)
        nt, h, w = imgs.shape
        nr = radius.size
        gl = gaussian_laplace_2d(imgs, radius, axis=1)
        gl_max = max_pool_3d(gl, 3)
        gl_peak = gl == gl_max
        if mask is not None:
            gl_peak &= mask
        gl = ops.where(gl_peak, gl, -inf)
        gl_reshape = gl.reshape(nt * nr, h, w)
        idx = ops.argmax(gl_reshape, axis=0)
        gl_max = ops.take_along_axis(gl_reshape, idx[None, ...], axis=0)[0]
        t, r = idx // nr, ops.mod(idx, nr)
        return t, r, gl_max
