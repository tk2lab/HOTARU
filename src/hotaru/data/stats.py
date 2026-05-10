from logging import getLogger
from math import inf
from math import nan

import numpy as np
from keras import ops

from ..models import Model
from ..ops import neighbor
from ..saving import Config
from ..saving import Data
from ..saving import cached_getter
from ..typing import Array
from ..typing import Tensor
from .dataset import MovieDataset
from .imgs import MovieData

logger = getLogger(__name__)


class Stats(Data):
    avgt: Array
    avgx: Array
    std0: Array
    min0: Array
    max0: Array
    imin: Array
    imax: Array
    istd: Array
    icor: Array


class StatsCalculator(Model):
    @cached_getter(Stats)
    def get_stats(
        self,
        data: MovieData | Config,
        batch_size: int = -1,
        capacity: int = -1,
        compile_kwargs: dict | None = None,
        dataset_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
    ) -> Stats:
        compile_kwargs = compile_kwargs or {}
        dataset_kwargs = dataset_kwargs or {}
        fit_kwargs = fit_kwargs or {}

        self.compile(**compile_kwargs)

        data = MovieData.get(data)
        if not self.built:
            nt, h, w = data.shape
            if capacity == -1:
                capacity = nt
            self.build((capacity, h, w))
        self.reset(data.mask)

        dataset = MovieDataset(data, batch_size, **dataset_kwargs)
        fit_kwargs.setdefault('shuffle', False)
        super().fit(dataset, **fit_kwargs)

        mask = self.mask.numpy()

        min0 = np.min(self.min0.numpy())
        max0 = np.max(self.min0.numpy())

        avgt = self.avgt.value[:-1]
        nt = self.avgt.shape[0] - 1

        avgx = np.where(mask, self.sumi.numpy() / nt, nan)
        varx = self.sqi.numpy() / nt - np.square(avgx)
        std0 = np.sqrt(np.nanmean(varx))

        stdx = np.sqrt(varx)
        avgn = np.where(mask, self.sumn.numpy() / nt, nan)
        stdn = np.sqrt(self.sqn.numpy() / nt - np.square(avgn))

        imin = (self.imin.numpy() - avgx) / std0
        imax = (self.imax.numpy() - avgx) / std0

        istd = np.where(mask, stdx / std0, nan)
        icor = np.where(mask, (self.cor.numpy() / nt - avgx * avgn) / (stdx * stdn), nan)

        return Stats(avgt, avgx, std0, min0, max0, imin, imax, istd, icor)

    def build(self, input_shape) -> None:
        nt, h, w = input_shape

        self.mask = self.add_weight(shape=(h, w), dtype='uint8', initializer='ones')
        self.avgt = self.add_weight(shape=(nt + 1,), dtype='float32', initializer='zeros')
        for key in ('sumi', 'sumn', 'sqi', 'sqn', 'cor'):
            setattr(self, key, self.add_weight(shape=(h, w), dtype='float32', initializer='zeros'))
        for key in ('min0', 'imin'):
            setattr(self, key, self.add_weight(shape=(h, w), dtype='float32', initializer='zeros'))
        for key in ('max0', 'imax'):
            setattr(self, key, self.add_weight(shape=(h, w), dtype='float32', initializer='zeros'))

        super().build(input_shape)

    def reset(self, mask: Array) -> None:
        self.mask.assign(mask)
        for key in ('sumi', 'sumn', 'sqi', 'sqn', 'cor'):
            v = getattr(self, key)
            v.assign(ops.zeros_like(v, 'float32'))
        for key in ('min0', 'imin'):
            v = getattr(self, key)
            v.assign(ops.full_like(v, +inf, 'float32'))
        for key in ('max0', 'imax'):
            v = getattr(self, key)
            v.assign(ops.full_like(v, -inf, 'float32'))

    def custom_train_step(self, ts: Tensor, imgs: Tensor) -> dict:
        imgs = ops.cast(imgs, 'float32')
        unpad = ops.where(ts[:, None, None] >= 0, imgs, nan)
        masked = ops.where(self.mask.value, unpad, nan)
        avgt, sumi, sqi, sumn, sqn, cor, min0, max0, imin, imax = self(masked)
        self.sumi.assign_add(sumi)
        self.sumn.assign_add(sumn)
        self.sqi.assign_add(sqi)
        self.sqn.assign_add(sqn)
        self.cor.assign_add(cor)
        self.min0.assign(ops.minimum(self.min0, min0))
        self.max0.assign(ops.maximum(self.min0, max0))
        self.imin.assign(ops.minimum(self.imin, imin))
        self.imax.assign(ops.maximum(self.imin, imax))
        self.avgt.assign(ops.scatter_update(self.avgt, ts[:, None], avgt))
        return {}

    def call(self, masked: Tensor) -> tuple[Tensor, ...]:
        avgti = ops.nanmean(masked, axis=(1, 2))
        diff = masked - avgti[..., None, None]
        neig = ops.where(ops.isnan(masked), nan, neighbor(ops.nan_to_num(diff, nan=0)))

        sumi = ops.nansum(diff, axis=0)
        sumn = ops.nansum(neig, axis=0)
        sqi = ops.nansum(ops.square(diff), axis=0)
        sqn = ops.nansum(ops.square(neig), axis=0)
        cor = ops.nansum(diff * neig, axis=0)

        min0 = ops.nanmin(masked, axis=0)
        max0 = ops.nanmax(masked, axis=0)
        imin = ops.nanmin(diff, axis=0)
        imax = ops.nanmax(diff, axis=0)

        return avgti, sumi, sqi, sumn, sqn, cor, min0, max0, imin, imax
