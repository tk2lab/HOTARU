from logging import getLogger
from math import inf
from math import nan

from keras import ops
from keras.callbacks import History
from keras.initializers import Constant

from ..models import Model
from ..ops import neighbor
from ..typing import Array
from ..typing import Tensor
from .dataset import CalciumImagingDataset

logger = getLogger(__name__)


class StatsCalculator(Model):
    def __init__(self, imgs: Array, mask: Array, **kwargs):
        super().__init__(**kwargs)
        self.imgs = imgs

        nt, h, w = self.imgs.shape
        self.mask = self.add_weight((h, w), dtype='uint8', initializer=mask)
        self.avgt = self.add_weight((nt + 1,), initializer='zeros')
        for key in ('sumi', 'sumn', 'sqi', 'sqn', 'cor'):
            setattr(self, key, self.add_weight((h, w), initializer='zeros'))
        for key in ('min0', 'imin'):
            setattr(self, key, self.add_weight((h, w), initializer=Constant(inf)))
        for key in ('max0', 'imax'):
            setattr(self, key, self.add_weight((h, w), initializer=Constant(-inf)))
        self._build_at_init()

    def fit(self, batch_size: int, **kwargs) -> History:
        kwargs.setdefault('shuffle', False)
        dataset_kwargs = kwargs.pop('dataset_kwargs', {})
        dataset = CalciumImagingDataset(self.imgs, batch_size, **dataset_kwargs)
        return super().fit(dataset, **kwargs)

    def custom_train_step(self, data) -> dict:
        ts, imgs = data
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
        diff = masked - avgti[:, None, None]
        neig = ops.where(ops.isfinite(diff), neighbor(ops.nan_to_num(diff, nan=0)), nan)

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

    def post_fit(self) -> tuple[Tensor, ...]:
        mask = self.mask.value

        min0 = ops.min(self.min0.value)
        max0 = ops.max(self.max0.value)

        avgt = self.avgt.value[:-1]
        nt = avgt.size

        avgx = ops.where(mask, self.sumi.value / nt, nan)
        varx = self.sqi.value / nt - ops.square(avgx)
        std0 = ops.sqrt(ops.nanmean(varx))

        stdx = ops.sqrt(varx)
        avgn = ops.where(mask, self.sumn.value / nt, nan)
        stdn = ops.sqrt(self.sqn.value / nt - ops.square(avgn))

        imin = (self.imin.value - avgx) / std0
        imax = (self.imax.value - avgx) / std0

        istd = ops.where(mask, stdx / std0, nan)
        icor = ops.where(mask, (self.cor.value / nt - avgx * avgn) / (stdx * stdn), nan)

        return avgt, avgx, std0, min0, max0, imin, imax, istd, icor
