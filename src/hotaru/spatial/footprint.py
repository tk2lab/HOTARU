from logging import getLogger
from math import nan

import h5py
import numpy as np
from keras import ops
from scipy.ndimage import grey_closing

from ..data import CalciumImagingDataWithStats as ImagingData
from ..models import Layer
from ..typing import Array
from ..typing import DType
from ..typing import Shape
from .clip import FootprintClipper
from .peaklist import PeakList

logger = getLogger(__name__)


class Footprints(Layer):
    def build(self, input_shape) -> None:
        num, h, w = input_shape
        self.segs = self.add_weight((num + 1, h, w), name='segs')
        super().build(input_shape)

    def from_peaklist(self, data: ImagingData, peaklist: PeakList, **kwargs):
        _, h, w = data.shape
        num = peaklist.size
        self.build((num, h, w))
        clipper = FootprintClipper(data, peaklist, self.segs)
        clipper.compile(**kwargs.pop('compile_kwargs', {}))
        clipper.fit_multi(**kwargs)
        #footprints = grey_closing(self.segs.numpy()[:-1], (1, 10, 10))
        #glist = np.sum(footprints, axis=(1, 2))
        #idx = np.flip(np.argsort(glist))
        #self.segs.assign(ops.pad(footprints[idx], ((0, 1), (0, 0), (0, 0)), constant_values=nan))
        #self._ys = peaklist.ylist
        #self._xs = peaklist.xlist
        #self._rs = peaklist.rlist
        #self._gs = peaklist.glist

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
        num, h, w = self.segs.shape
        return num - 1, h, w

    @property
    def dtype(self) -> DType:
        return self.segs.dtype

    @property
    def rs(self) -> Array:
        return self._rs

    @property
    def ys(self) -> Array:
        return self._ys

    @property
    def xs(self) -> Array:
        return self._xs

    @property
    def core(self):
        return clip_core(self.imgs.numpy(), self.ys, self.xs)[0]


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
