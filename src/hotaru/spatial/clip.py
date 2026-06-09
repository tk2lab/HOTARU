from logging import getLogger
from math import ceil
from math import log2
from math import nan

import numpy as np
from keras import Variable
from keras import ops
from keras.callbacks import History
from keras.utils import PyDataset

from ..data import CalciumImagingDataWithStats as ImagingData
from ..models import Model
from ..ops import laplace_2d
from ..ops import laplacian_of_gaussian_kernel
from ..typing import Array
from .segment import get_segment_mask

logger = getLogger(__name__)


class FootprintClipper(Model):
    def __init__(self, data: ImagingData, out_segs: Variable, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.segs = out_segs
        self._build_at_init()

    def fit_multi(
        self,
        tlist: Array,
        rlist: Array,
        ylist: Array,
        xlist: Array,
        batch_size: int,
        kind: str = 'segs',
        **kwargs,
    ) -> list[History]:
        dataset_args = self.data.imgs, tlist, rlist, ylist, xlist, batch_size
        dataset_kwargs = kwargs.pop('dataset_kwargs', {})
        history = []
        for radius in np.unique(rlist):
            desc = f'clip ({kind}; r={radius:.3f})'
            cumsum = {'active': 0}
            dataset = FrameWithPeakDataset(*dataset_args, radius=radius, **dataset_kwargs)
            history.append(super().fit(dataset, **kwargs, desc=desc, cumsum=cumsum))
        return history

    def custom_train_step(self, data) -> dict:
        ((peak_id, imgs, kernel0, kernel2, ts, ys, xs),) = data
        imgs = self.data.normalize(ts, imgs)
        segs = self(imgs, kernel0, kernel2, ys, xs)
        self.segs.assign(ops.scatter_update(self.segs, peak_id[:, None, None], segs))
        n_active = ops.count_nonzero(ops.any(segs > 0, axis=(1, 2)))
        return {'active': n_active}

    def call(self, img, kernel0, kernel2, y, x):
        g = laplace_2d(img, kernel0, kernel2)
        seg_mask = ops.vectorized_map(lambda args: get_segment_mask(*args), (g, y, x))
        val = ops.where(seg_mask, g, nan)
        val -= ops.nanmin(val, axis=(1, 2), keepdims=True)
        val = ops.where(seg_mask, val, 0)
        return val


class FrameWithPeakDataset(PyDataset):
    def __init__(
        self,
        imgs: Array,
        tlist: Array,
        rlist: Array,
        ylist: Array,
        xlist: Array,
        batch_size: int,
        radius: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nd = max(32, 2 ** ceil(log2(4 * radius)))
        (peak_id,) = np.nonzero(rlist == radius)
        self.kernel0, self.kernel2 = laplacian_of_gaussian_kernel(radius, nd=nd)
        self.imgs = imgs
        self.peak_id = peak_id
        self.ts = tlist[peak_id]
        self.ys = ylist[peak_id]
        self.xs = xlist[peak_id]
        self.batch_size = batch_size

    def on_epoch_end(self):
        pass

    def __len__(self) -> int:
        return (self.peak_id.size + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int):
        s = self.batch_size * index
        e = s + self.batch_size
        diff = max(e - self.ts.size, 0)
        peak_id = np.pad(self.peak_id[s:e], ((0, diff)), constant_values=self.peak_id.size)
        ts = np.pad(self.ts[s:e], ((0, diff)))
        ys = np.pad(self.ys[s:e], ((0, diff)))
        xs = np.pad(self.xs[s:e], ((0, diff)))
        imgs = self.imgs[ts]
        return ((peak_id, imgs, self.kernel0, self.kernel2, ts, ys, xs),)
