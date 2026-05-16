from logging import getLogger
from math import nan

import numpy as np
from keras import ops
from keras.utils import PyDataset
from scipy.ndimage import grey_closing

from ..data import MovieData
from ..data import MovieWithStats
from ..models import Model
from ..ops import gaussian_laplace_2d
from ..saving import Config
from ..saving import PathLike
from ..saving import cached_getter
from .footprint import Footprints
from .reduce import PeakList
from .segment import get_segment_mask

logger = getLogger(__name__)


class MovieAndPeaksDataset(PyDataset):
    def __init__(self, data: MovieData, peaks: PeakList, radius: float, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        (peak_id,) = np.nonzero(peaks.rlist == radius)
        self.peak_id = peak_id
        self.imgs = data.imgs
        self.ts = peaks.tlist[peak_id]
        self.ys = peaks.ylist[peak_id]
        self.xs = peaks.xlist[peak_id]
        self.batch_size = batch_size

    def on_epoch_end(self):
        pass

    def __len__(self) -> int:
        return (self.peak_id.size + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int):
        s = self.batch_size * index
        e = s + self.batch_size
        diff = max(e - self.ts.size, 0)
        peak_id = np.pad(self.peak_id[s:e], ((0, diff)), constant_values=-1)
        imgs = np.pad(self.imgs[self.ts[s:e]], ((0, diff), (0, 0), (0, 0)))
        ys = np.pad(self.ys[s:e], ((0, diff)))
        xs = np.pad(self.xs[s:e], ((0, diff)))
        return ((peak_id, imgs, ys, xs),)


class FootprintClipper(Model):
    @cached_getter(Footprints)
    def get_footprints(
        self,
        data: MovieWithStats | Config,
        peaks: PeakList | PathLike,
        batch_size: int,
        capacity: int = -1,
        compile_kwargs: dict | None = None,
        dataset_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
    ) -> Footprints:
        compile_kwargs = compile_kwargs or {}
        dataset_kwargs = dataset_kwargs or {}
        fit_kwargs = fit_kwargs or {}

        data = MovieWithStats.get(data)
        if not isinstance(peaks, PeakList):
            peaks = PeakList.load(peaks)

        num = peaks.size
        if not self.built:
            _, h, w = data.shape
            if capacity == -1:
                capacity = num
            self.build((capacity, h, w))
        self.avgx.assign(data.stats.avgx)
        self.std0.assign(data.stats.std0)

        for radius in np.unique(peaks.rlist):
            self.radius = radius
            self.compile(**compile_kwargs)
            dataset = MovieAndPeaksDataset(data, peaks, radius, batch_size, **dataset_kwargs)
            super().fit(dataset, **fit_kwargs)

        footprints = grey_closing(self.segs.numpy()[:num], (1, 10, 10))
        glist = np.sum(footprints, axis=(1, 2))
        idx = np.flip(np.argsort(glist))
        return Footprints(footprints[idx], peaks.ylist[idx], peaks.xlist[idx])

    def build(self, input_shape) -> None:
        num, h, w = input_shape
        self.avgx = self.add_weight(shape=(h, w), dtype='float32', initializer='zeros')
        self.std0 = self.add_weight(shape=(), dtype='float32', initializer='ones')
        self.segs = self.add_weight(shape=(num + 1, h, w), dtype='float32')
        super().build(input_shape)

    def custom_train_step(self, data) -> dict:
        ((peak_id, imgs, y, x),) = data
        imgs = (ops.cast(imgs, 'float32') - self.avgx) / self.std0
        imgs = ops.where(peak_id[:, None, None] >= 0, imgs, nan)
        segs = self(imgs, y, x)
        self.segs.assign(ops.scatter_update(self.segs, peak_id[:, None, None], segs))
        return {}

    def call(self, img, y, x):
        g = gaussian_laplace_2d(img, self.radius)
        seg_mask = ops.vectorized_map(_get_segment_mask, (g, y, x))
        val = ops.where(seg_mask, g, nan)
        val -= ops.nanmin(val, axis=(1, 2), keepdims=True)
        val = ops.where(seg_mask, val, 0)
        return val


def _get_segment_mask(args):
    return get_segment_mask(*args)
