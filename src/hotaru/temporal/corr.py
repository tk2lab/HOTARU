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
        footprint: Tensor,
        batch_size: int,
        compile_kwargs: dict | None = None,
        dataset_kwargs: dict | None = None,
        **kwargs,
    ) -> Tensor:
        compile_kwargs = compile_kwargs or {}
        dataset_kwargs = dataset_kwargs or {}

        nk = footprint.shape[0]
        if not self.built:
            nt, h, w = data.shape
            self.build((nk, nt, h * w))

        self.compile(**compile_kwargs)

        self.avgt.assign(data.stats.avgt)
        self.avgx.assign(data.stats.avgx.ravel())
        self.std0.assign(data.stats.std0)

        if (gap := self.spatial_comp.shape[0] - footprint.shape[0]) > 0:
            footprint = ops.pad(footprint, ((0, gap), (0, 0)))
        self.spatial_comp.assign(footprint)

        dataset = MovieDataset(data, batch_size, **dataset_kwargs)
        self.fit(dataset, **kwargs)

        return self.spatial_corr.value[:nk]

    def build(self, input_shape) -> None:
        nk, nt, nx = input_shape
        capacity = nk if self.capacity == -1 else self.capacity
        self.avgt = self.add_weight((nt,), trainable=False)
        self.avgx = self.add_weight((nx,), trainable=False)
        self.std0 = self.add_weight((), trainable=False)
        self.spatial_comp = self.add_weight((capacity, nx))
        self.spatial_corr = self.add_weight((capacity, nt))
        super().build(input_shape)

    def custom_train_step(self, data) -> dict:
        ts, imgs = data
        nt, h, w = imgs.shape
        imgs = ops.reshape(imgs, (nt, h * w))
        imgs = (ops.cast(imgs, 'float32') - self.avgt[ts, None] - self.avgx) / self.std0
        src = (self.spatial_comp.value @ imgs.T) / h / w
        dst = self.spatial_corr
        dst.assign(ops.scatter_update(dst.value.T, ts[:, None], src.T).T)
        return {}
