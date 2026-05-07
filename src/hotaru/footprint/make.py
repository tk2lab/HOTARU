from dataclasses import dataclass
from logging import getLogger
from math import nan

import numpy as np
from keras import ops
from keras.callbacks import History
from keras.utils import PyDataset
from keras.utils import unpack_x_y_sample_weight
from scipy.ndimage import grey_closing

from ..io import MovieData
from ..model import Model
from ..ops import gaussian_laplace_2d_single
from ..saving import Config
from ..typing import Array
from .radius import Radius
from .segment import get_segment_mask

logger = getLogger(__name__)


@dataclass
class PeakList:
    radius: Radius
    ts: Array
    ri: Array
    ys: Array
    xs: Array
    gs: Array


class MakerDataset(PyDataset):
    def __init__(self, data: MovieData, peaks: PeakList, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        ri_list = np.unique(peaks.ri)
        self.batch = []
        for ri in ri_list:
            radius = peaks.radius[ri]
            peakid = np.nonzero(peaks.ri == ri)[0]
            ts = peaks.ts[peakid]
            ys = peaks.ys[peakid]
            xs = peaks.xs[peakid]
            img = data.data[ts]
            for s in range(0, ts.size, batch_size):
                e = s + batch_size
                self.batch.append((peakid[s:e], img[s:e], radius, ys[s:e], xs[s:e]))

    def on_epoch_end(self):
        pass

    def __len__(self) -> int:
        return len(self.batch)

    def __getitem__(self, index: int):
        return self.batch[index], ()


class FootprintMaker(Model):
    def __init__(self, capacity: int, **kwargs):
        super().__init__(**kwargs)
        self.capacity = capacity

    def fit(self, data: MovieData | Config, peaks: PeakList, batch_size: int, **kwargs) -> History:
        data = MovieData.get(data)
        dataset = MakerDataset(data, peaks, batch_size)
        history = super().fit(dataset, **kwargs)
        return history

    def get_segments(self):
        segs = ops.convert_to_numpy(self.segs.value)
        return grey_closing(segs, (1, 10, 10))

    def build(self, input_shapes) -> None:
        _, (_, h, w), _, _, _ = input_shapes
        self.segs = self.add_weight(shape=(self.capacity, h, w), dtype='float32')
        super().build(input_shapes)

    def custom_train_step(self, data) -> dict:
        (peakid, img, r, y, x), _y, _sample_weight = unpack_x_y_sample_weight(data)
        seg = self(img, r, y, x)
        self.segs.assign(ops.scatter_update(self.segs, (peakid, 0, 0), seg))
        return {}

    def call(self, img, r, y, x):
        g = gaussian_laplace_2d_single(img, r)
        seg = ops.vectorized_map(get_segment_mask, (g, y, x))
        val = ops.where(seg, g, nan)
        dmin = ops.nanmin(val, axis=(1, 2), keepdims=True)
        dmax = ops.nanmax(val, axis=(1, 2), keepdims=True)
        out = ops.where(seg, (val - dmin) / (dmax - dmin), 0)
        return out
