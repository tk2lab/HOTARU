from logging import getLogger

import h5py
import numpy as np

from ..saving import Data
from ..saving import PathLike
from ..typing import Array
from ..typing import DType
from ..typing import Shape

logger = getLogger(__name__)


class Footprints(Data):
    data: Array
    y: Array
    x: Array

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def dtype(self) -> DType:
        return self.data.dtype

    @property
    def core(self):
        return clip_core(self.data, self.y, self.x)[0]

    def save(self, path: PathLike) -> None:
        layout = h5py.VirtualLayout(self.shape, self.dtype)
        core, clip, pad = clip_core(self.data, self.y, self.x)
        num, core_h, core_w = core.shape
        with h5py.File(path, 'w') as db:
            core_ds = db.create_dataset('core', data=core)
            vs = h5py.VirtualSource(core_ds)
            for i in range(num):
                t_clip, b_clip, l_clip, r_clip = clip[i]
                pad_t, pad_b, pad_l, pad_r = pad[i]
                vs_t, vs_b = pad_t, core_h - pad_b
                vs_l, vs_r = pad_l, core_w - pad_r
                vs_clip = vs[i, vs_t:vs_b, vs_l:vs_r]
                layout[i, t_clip:b_clip, l_clip:r_clip] = vs_clip
            db.create_virtual_dataset('data', layout)
            db.create_dataset('y', data=self.y)
            db.create_dataset('x', data=self.x)


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
