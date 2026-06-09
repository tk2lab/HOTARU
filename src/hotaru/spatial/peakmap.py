from logging import getLogger
from math import inf

from keras import ops

from ..data import CalciumImagingDataWithStats as ImagingData
from ..models import Layer
from ..models import Model
from ..ops import gaussian_laplace_2d_multi
from ..ops import local_peaks
from ..saving import Config
from ..typing import Shape
from ..typing import Tensor
from .radius import Radius

logger = getLogger(__name__)


class PeakMap(Layer):
    def __init__(self, radius: Radius | Config, shape: Shape, **kwargs):
        super().__init__(**kwargs)
        self.radius = Radius.get(radius)
        self.tmap = self.add_weight(shape, dtype='int32', name='tmap')
        self.rmap = self.add_weight(shape, dtype='int32', name='rmap')
        self.gmap = self.add_weight(shape, name='gmap')
        self._build_at_init()

    def get_config(self) -> Config:
        return {**super().get_config(), 'radius': self.radius.tolist(), 'shape': self.tmap.shape}

    @classmethod
    def generate(cls, data: ImagingData, radius: Radius | Config, **kwargs):
        _, h, w = data.shape
        obj = PeakMap(radius, (h, w), **kwargs.pop('layer', {}))
        fit_kwargs = dict(kwargs.pop('fit', {}))
        if 'desc' in kwargs:
            fit_kwargs['desc'] = kwargs.pop('desc')
        calculator = PeakMapCalculator(data, obj, **kwargs)
        calculator.calc(**fit_kwargs)
        return obj


class PeakMapCalculator(Model):
    def __init__(
        self,
        data: ImagingData,
        out: PeakMap,
        pool_hsize: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data = data
        self.hsize = pool_hsize
        self.out = out
        self._build_at_init()

    def calc(self, batch_size: int, **kwargs) -> None:
        if not self.compiled or 'compile_kwargs' in kwargs:
            self.compile(**kwargs.pop('compile_kwargs', {}))
        kwargs.setdefault('shuffle', False)
        self.reset()
        dataset = self.data.dataset(batch_size, self.hsize, **kwargs.pop('dataset_kwargs', {}))
        self.fit(dataset, **kwargs)

    def reset(self) -> None:
        self.out.gmap.assign(ops.full(self.out.gmap.shape, -inf, 'float32'))

    def custom_train_step(self, data) -> dict:
        ts, imgs = data
        imgs = self.data.normalize(ts, imgs)
        i, r, g = self(imgs)
        t = ts[i]
        cond = g > self.out.gmap.value
        self.out.tmap.assign(ops.where(cond, t, self.out.tmap.value))
        self.out.rmap.assign(ops.where(cond, r, self.out.rmap.value))
        self.out.gmap.assign(ops.where(cond, g, self.out.gmap.value))
        return {}

    def call(self, imgs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        imgs = ops.nan_to_num(imgs, nan=0)
        mask = ops.isfinite(self.data.avgx)
        radius = self.out.radius

        gl = gaussian_laplace_2d_multi(imgs, radius, axis=1)
        gl_peak = local_peaks(gl, 4, 1)
        glp_tr = ops.where(mask & gl_peak, gl, -inf)

        nt, nr, h, w = glp_tr.shape
        glp_flat = ops.reshape(glp_tr, (nt * nr, h, w))
        tr = ops.argmax(glp_flat, axis=0)
        t, r = tr // nr, tr % nr
        glp = ops.take_along_axis(glp_flat, tr[None, :, :], axis=0)[0, :, :]
        return t, r, glp
