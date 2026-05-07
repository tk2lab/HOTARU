from logging import getLogger

import numpy as np

from ..typing import Array

logger = getLogger(__name__)


def reduce_peaks_simple(
    yxrv: tuple[Array, Array, Array, Array],
    min_radius: float,
    max_radius: float,
    min_distance_ratio: float,
    old_bg: list | None = None,
) -> tuple[Array, Array, Array]:
    if old_bg is None:
        old_bg = []

    ys, xs, rs, vs = (np.ravel(a) for a in yxrv)
    n = np.count_nonzero(np.isfinite(vs))
    flg = np.flip(np.argsort(vs))[:n]
    cell = []
    bg = []
    remove = []
    while flg.size > 0:
        i, flg = flg[0], flg[1:]
        r0 = rs[i]
        if r0 >= min_radius:
            y0, x0 = ys[i], xs[i]
            if i in old_bg or r0 > max_radius:
                yb, xb = ys[bg], xs[bg]
                distance = np.hypot(xb - x0, yb - y0) / r0
                if not bg or np.all(distance >= min_distance_ratio):
                    bg.append(i)
                else:
                    remove.append(i)
            else:
                yc, xc = ys[cell], xs[cell]
                distance = np.hypot(xc - x0, yc - y0) / r0
                if not cell or np.all(distance >= min_distance_ratio):
                    y1, x1 = ys[flg], xs[flg]
                    dist2 = np.hypot(x1 - x0, y1 - y0) / r0
                    cond = dist2 > min_distance_ratio
                    if np.any(~cond):
                        remove += list(flg[~cond])
                    flg = flg[cond]
                    cell.append(i)
                else:
                    remove.append(i)
        else:
            remove.append(i)
    return np.array(cell, 'int32'), np.array(bg, 'int32'), np.array(remove, 'int32')
