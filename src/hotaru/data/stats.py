from logging import getLogger
from math import inf
from math import nan

from keras import ops
from keras.callbacks import History
from keras.initializers import Constant

from ..model import Model
from ..ops import neighbor
from ..saving import Config
from ..saving import Data
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
    def fit(self, data: MovieData | Config, batch_size: int = -1, **kwargs) -> History:
        if self.built:
            raise RuntimeError()
        data = MovieData.get(data)
        self.build(data.shape)
        self.mask.assign(data.mask)
        dataset = MovieDataset(data, batch_size)
        kwargs.setdefault('shuffle', False)
        return super().fit(dataset, **kwargs)

    def get_stats(self) -> Stats:
        mask = self.mask.value
        nt = self.avgt.shape[0] - 1

        min0 = ops.min(self.min0)
        max0 = ops.max(self.min0)

        avgt = self.avgt.value[:-1]

        avgx = ops.where(mask, self.sumi / nt, nan)
        varx = self.sqi / nt - ops.square(avgx)
        std0 = ops.sqrt(ops.nanmean(varx))

        stdx = ops.sqrt(varx)
        avgn = ops.where(mask, self.sumn / nt, nan)
        stdn = ops.sqrt(self.sqn / nt - ops.square(avgn))

        imin = (self.imin - avgx) / std0
        imax = (self.imax - avgx) / std0

        istd = ops.where(mask, stdx / std0, nan)
        icor = ops.where(mask, (self.cor / nt - avgx * avgn) / (stdx * stdn), nan)

        stats = avgt, avgx, std0, min0, max0, imin, imax, istd, icor
        return Stats(*(ops.convert_to_numpy(val) for val in stats))

    def build(self, input_shape) -> None:
        pinf = Constant(inf)
        ninf = Constant(-inf)
        nt, h, w = input_shape

        self.mask = self.add_weight(shape=(h, w), dtype='uint8', initializer='ones')
        self.avgt = self.add_weight(shape=(nt + 1,), dtype='float32', initializer='zeros')
        for key in ('sumi', 'sumn', 'sqi', 'sqn', 'cor'):
            setattr(self, key, self.add_weight(shape=(h, w), dtype='float32', initializer='zeros'))
        for key in ('min0', 'imin'):
            setattr(self, key, self.add_weight(shape=(h, w), dtype='float32', initializer=pinf))
        for key in ('max0', 'imax'):
            setattr(self, key, self.add_weight(shape=(h, w), dtype='float32', initializer=ninf))

        super().build(input_shape)

    def custom_train_step(self, ts: Tensor, imgs: Tensor) -> dict:
        imgs = ops.cast(imgs, 'float32')
        unpad = ops.where(ts[:, None, None] >= 0, imgs, nan)
        masked = ops.where(self.mask.value, unpad, nan)
        avgt, sumi, sqi, sumn, sqn, cor, min0, max0, imin, imax = self(masked)
        self.sumi.assign(self.sumi + sumi)
        self.sumn.assign(self.sumn + sumn)
        self.sqi.assign(self.sqi + sqi)
        self.sqn.assign(self.sqn + sqn)
        self.cor.assign(self.cor + cor)
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
