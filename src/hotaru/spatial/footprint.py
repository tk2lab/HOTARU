from logging import getLogger
from math import nan

import h5py
import numpy as np
from keras import ops
from scipy.ndimage import maximum_position

from ..data import CalciumImagingDataWithStats as ImagingData
from ..models import Layer
from ..saving import Config
from ..typing import Array
from ..typing import DType
from ..typing import Shape
from .clip import FootprintClipper
from .peaklist import PeakList

logger = getLogger(__name__)


class Footprints(Layer):
    def __init__(self, segs_or_shape, **kwargs):
        super().__init__(**kwargs)
        match segs_or_shape:
            case np.ndarray() as segs:
                self.segs = self.add_weight(segs.shape, initializer=segs, name='segs')
            case shape:
                self.segs = self.add_weight(shape, name='segs')
        self._build_at_init()

    def get_config(self) -> Config:
        return {'segs_or_shape': self.segs.shape, **super().get_config()}

    @classmethod
    def from_peaklist(cls, data: ImagingData, peaklist: PeakList, **kwargs):
        _, h, w = data.shape
        num = peaklist.size
        obj = Footprints((num, h, w))
        clipper = FootprintClipper(data, peaklist, obj.segs)
        clipper.compile(**kwargs.pop('compile_kwargs', {}))
        clipper.fit_multi(**kwargs)
        return obj

    def save_own_weights(self, store) -> None:
        layout = h5py.VirtualLayout(self.shape, self.dtype)
        core, clip, pad = clip_core(self.imgs.numpy(), self.ys, self.xs)
        num, core_h, core_w = core.shape
        core_ds = store.create_dataset('core', data=core)
        vs = h5py.VirtualSource(core_ds)
        for i in range(num):
            t_clip, b_clip, l_clip, r_clip = clip[i]
            pad_t, pad_b, pad_l, pad_r = pad[i]
            vs_t, vs_b = pad_t, core_h - pad_b
            vs_l, vs_r = pad_l, core_w - pad_r
            vs_clip = vs[i, vs_t:vs_b, vs_l:vs_r]
            layout[i, t_clip:b_clip, l_clip:r_clip] = vs_clip
        store.create_virtual_dataset('segs', layout)

    def load_own_weights(self, store) -> None:
        segs = store['segs'][...]
        self.segs.assign(ops.pad(segs, ((0, 1), (0, 0), (0, 0)), constant_values=nan))

    @property
    def shape(self) -> Shape:
        return self.segs.shape

    @property
    def dtype(self) -> DType:
        return self.segs.dtype

    @property
    def peaks(self) -> Array:
        fn = np.vectorize(lambda x: np.stack(maximum_position(x)), signature='(x,y)->(k)')
        return fn(self.segs.numpy())

    @property
    def core(self) -> Array:
        ys, xs = self.peaks.T
        return clip_core(self.imgs.numpy(), ys, xs)[0]


def get_bounds(x):
    *_, size = x.shape
    xs = np.arange(size)
    xmin = np.min(np.where(x, xs, size), axis=-1)
    xmax = np.max(np.where(x, xs, -1), axis=-1)
    return xmin, xmax


def clip_core(data, y, x):
    num, h, w = data.shape
    dtype = data.dtype
    mask = data > 0

    x0, x1 = get_bounds(np.any(mask, axis=1))
    y0, y1 = get_bounds(np.any(mask, axis=2))

    dx0 = int(np.max(x - x0))
    dx1 = int(np.max(x1 - x))
    dy0 = int(np.max(y - y0))
    dy1 = int(np.max(y1 - y))
    core_h, core_w = dy0 + dy1 + 1, dx0 + dx1 + 1

    core_t, core_l = y - dy0, x - dx0
    core_b, core_r = core_t + core_h, core_l + core_w

    clip = np.stack(
        [
            np.maximum(0, core_t),
            np.minimum(h, core_b),
            np.maximum(0, core_l),
            np.minimum(w, core_r),
        ],
        axis=-1,
    )
    pad = np.stack(
        [
            np.maximum(0, -core_t),
            np.maximum(0, core_b - h),
            np.maximum(0, -core_l),
            np.maximum(0, core_r - w),
        ],
        axis=-1,
    )

    core = np.empty((num, core_h, core_w), dtype)
    for i in range(num):
        t_clip, b_clip, l_clip, r_clip = clip[i]
        pad_t, pad_b, pad_l, pad_r = pad[i]
        roi = data[i, t_clip:b_clip, l_clip:r_clip]
        core[i] = np.pad(roi, ((pad_t, pad_b), (pad_l, pad_r)))

    return core, clip, pad
