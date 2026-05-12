from keras import ops

from ..data import MovieDataset
from ..data import MovieWithStats
from ..models import Model
from ..saving import Data
from ..saving import DataDict
from ..saving import cached_getter
from ..typing import Array


class CorrData(Data):
    data: Array


class CorrCalculator(Model):
    def __init__(self, props, spatial_comp, spatial_corr, **kwargs):
        super().__init__(**kwargs)
        self.props = props
        self.spatial_comp = spatial_comp
        self.spatial_corr = spatial_corr

    @cached_getter(DataDict)
    def calc_corr(
        self,
        data: MovieWithStats,
        batch_size: int,
        compile_kwargs: dict | None = None,
        dataset_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
    ):
        compile_kwargs = compile_kwargs or {}
        dataset_kwargs = dataset_kwargs or {}
        fit_kwargs = fit_kwargs or {}

        self.compile(**compile_kwargs)

        if not self.built:
            self.build(data.shape)

        self.avgt.assign(data.stats.avgt)
        self.avgx.assign(data.stats.avgx.ravel())
        self.std0.assign(data.stats.std0)

        dataset = MovieDataset(data, batch_size, **dataset_kwargs)
        fit_kwargs.setdefault('shuffle', False)
        self.fit(dataset, **fit_kwargs)

        return [CorrData(d.numpy()) for d in self.spatial_corr]

    def build(self, input_shape) -> None:
        nt, h, w = input_shape
        self.avgt = self.add_weight((nt,), trainable=False)
        self.avgx = self.add_weight((h * w,), trainable=False)
        self.std0 = self.add_weight((), trainable=False)
        super().build(input_shape)

    def custom_train_step(self, data) -> dict:
        ts, imgs = data
        nt, h, w = imgs.shape
        imgs = ops.reshape(imgs, (nt, h * w))
        imgs = (ops.cast(imgs, 'float32') - self.avgt[ts, None] - self.avgx) / self.std0
        for i, ai in enumerate(self.spatial_comp):
            src = (ai.value @ imgs.T) / h / w
            dst = self.spatial_corr[i]
            dst.assign(ops.scatter_update(dst.value.T, ts[:, None], src.T).T)
        return {}
