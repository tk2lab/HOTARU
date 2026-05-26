from logging import getLogger
from math import inf

from keras import ops
from keras.callbacks import History

from ..data import CalciumImagingDataWithStats
from ..models import Layer
from ..models import Model
from ..ops import gaussian_laplace_2d_multi
from ..ops import max_pool_3d
from ..saving import Config
from ..typing import Tensor
from .radius import Radius

logger = getLogger(__name__)


class PeakMap(Layer):
    def __init__(self, radius: Radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = Radius.get(radius)

    def get_config(self) -> Config:
        return {'radius': self.radius.get_config(), **super().get_config()}

    def build(self, input_shape) -> None:
        _, h, w = input_shape
        self.tmap = self.add_weight((h, w), dtype='int32', name='tmap')
        self.rmap = self.add_weight((h, w), dtype='int32', name='rmap')
        self.gmap = self.add_weight((h, w), name='gmap')
        super().build(input_shape)

    def calc(self, data: CalciumImagingDataWithStats, **kwargs):
        self.build(data.shape)
        compile_kwargs = kwargs.pop('compile_kwargs', {})
        calc = PeakMapCalculator(data, self.radius, self)
        calc.compile(**compile_kwargs)
        calc.fit(**kwargs)


class PeakMapCalculator(Model):
    def __init__(
        self,
        data: CalciumImagingDataWithStats,
        radius: Radius,
        out: PeakMap,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.radius = radius
        self.out = out
        super()._build_at_init()

    def fit(self, batch_size: int, **kwargs) -> History:
        dataset_kwargs = kwargs.pop('dataset_kwargs', {})
        dataset = self.data.dataset(batch_size, **dataset_kwargs)
        kwargs.setdefault('shuffle', False)
        self.out.gmap.assign(ops.full(self.out.gmap.shape, -inf, 'float32'))
        return super().fit(dataset, **kwargs)

    def custom_train_step(self, data) -> dict:
        ts, imgs = data
        imgs = self.data.normalize(ts, imgs)
        i, r, g = self(imgs)
        t = ops.take(ts, i)
        cond = g > self.out.gmap.value
        self.out.tmap.assign(ops.where(cond, t, self.out.tmap.value))
        self.out.rmap.assign(ops.where(cond, r, self.out.rmap.value))
        self.out.gmap.assign(ops.where(cond, g, self.out.gmap.value))
        return {}

    def call(self, imgs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        imgs = ops.nan_to_num(imgs, nan=0)
        mask = ops.isfinite(self.data.avgx)
        radius = self.radius

        *_, h, w = ops.shape(imgs)
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
