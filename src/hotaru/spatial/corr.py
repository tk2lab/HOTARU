from keras import ops

from ..data import MovieDataset
from ..data import MovieWithStats
from ..models import Model
from ..typing import Tensor


class CorrCalculator(Model):
    def __init__(self, capacity: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.capacity = capacity

    def calc(
        self,
        data: MovieWithStats,
        trace: Tensor,
        batch_size: int,
        compile_kwargs: dict | None = None,
        dataset_kwargs: dict | None = None,
        **kwargs,
    ) -> Tensor:
        compile_kwargs = compile_kwargs or {}
        dataset_kwargs = dataset_kwargs or {}

        nk = trace.shape[0]
        if not self.built:
            nt, h, w = data.shape
            self.build((nk, nt, h * w))

        self.compile(**compile_kwargs)

        self.avgt.assign(data.stats.avgt)
        self.avgx.assign(data.stats.avgx.ravel())
        self.std0.assign(data.stats.std0)

        if (gap := self.spatial_comp.shape[0] - trace.shape[0]) > 0:
            trace = ops.pad(trace, ((0, gap), (0, 0)))
        self.temporal_comp.assign(trace)
        self.temporal_corr.assign(ops.zeros_like(self.temporal_corr.value))

        dataset = MovieDataset(data, batch_size, **dataset_kwargs)
        self.fit(dataset, **kwargs)

        return self.spatial_corr.value[:nk]

    def build(self, input_shape) -> None:
        nk, nt, nx = input_shape
        capacity = nk if self.capacity == -1 else self.capacity
        args = {'trainable': False, 'initializer': 'zero'}
        self.avgt = self.add_weight((nt,), **args, name='avgt')
        self.avgx = self.add_weight((nx,), **args, name='avgx')
        self.std0 = self.add_weight((), **args, name='std0')
        self.temporal_comp = self.add_weight((capacity, nt), **args, name='comp')
        self.temporal_corr = self.add_weight((capacity, nx), **args, name='corr')
        super().build(input_shape)

    def custom_train_step(self, data) -> dict:
        ts, imgs = data
        nt, h, w = imgs.shape
        imgs = ops.reshape(imgs, (nt, h * w))
        imgs = (ops.cast(imgs, 'float32') - self.avgt[ts, None] - self.avgx) / self.std0
        self.temporal_corr.assign_add((self.temporal_comp.value @ imgs) / nt)
        return {}
