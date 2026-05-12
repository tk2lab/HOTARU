import numpy as np
from keras import ops

from ..data import MovieDataset
from ..data import MovieWithStats
from ..models import Model


class TemporalPrepare(Model):
    def __init__(self, updater, **kwargs):
        super().__init__(**kwargs)
        self.updater = updater

    def update(
        self,
        data: MovieWithStats,
        *footprints,
        batch_size: int,
        compile_kwargs: dict | None = None,
        dataset_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
    ):
        compile_kwargs = compile_kwargs or {}
        dataset_kwargs = dataset_kwargs or {}
        fit_kwargs = fit_kwargs or {}

        nt, h, w = data.shape
        nx = h * w
        nc = tuple(fp.data.shape[0] for fp in footprints)

        if not self.built:
            self.build((nt, nx, nc))

        self.avgt.assign(data.stats.avgt)
        self.avgx.assign(data.stats.avgx.ravel())
        self.std0.assign(data.stats.std0)

        self.updater.scale = float(nt) * float(nx)

        for i, ni in enumerate(nc):
            ai = footprints[i].data.reshape(ni, nx)
            ai /= ai.max()
            fac = self.updater.props.component_properties[i].temporal_factor(ai)
            self.updater.spatial_comp[i].assign(ai)
            self.updater.activity[i].regularizer.fac.assign(fac)

        for i, ai in enumerate(self.updater.spatial_comp):
            self.updater.spatial_mean[i].assign(ops.mean(ai, axis=1))
            for j, aj in enumerate(self.updater.spatial_comp[: i + 1]):
                sqrd = self.updater.spatial_sqrd[i][j]
                sqrd.assign(ai.value @ aj.value.T / nx)

        self.compile(**compile_kwargs)
        dataset = MovieDataset(data, batch_size, **dataset_kwargs)
        self.fit(dataset, **fit_kwargs)

    def build(self, input_shape) -> None:
        nt, nx, nc = input_shape
        if not self.updater.built:
            self.updater.build((nt, nx, nc))
        self.avgt = self.add_weight((nt,), trainable=False)
        self.avgx = self.add_weight((nx,), trainable=False)
        self.std0 = self.add_weight((), trainable=False)
        super().build(input_shape)

    def custom_train_step(self, data) -> dict:
        ts, imgs = data
        nt, h, w = imgs.shape
        nx = h * w
        imgs = ops.reshape(imgs, (nt, nx))
        imgs = (ops.cast(imgs, 'float32') - self.avgt[ts, None] - self.avgx) / self.std0
        for i, ai in enumerate(self.updater.spatial_comp):
            corr = self.updater.spatial_corr[i]
            corr.assign(ops.scatter_update(corr.value.T, ts[:, None], imgs @ ai.value.T).T / nx)
        return {}
