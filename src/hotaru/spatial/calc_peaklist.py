from logging import getLogger
from math import nan

import numpy as np
from tqdm import tqdm

# from ..typing import Array
from .radius import Radius

Array = np.typing.NDArray

logger = getLogger(__name__)


def calc_peaklist(
    radius: Radius,
    rimap: Array,
    tmap: Array,
    gmap: Array,
    min_distance_ratio: float,
    block_size: int,
):
    active = (rimap >= 1) & (rimap < radius.size - 2)
    rmap = np.where(active, radius[rimap], nan)
    gmap = np.where(active, gmap, nan)

    h, w = rmap.shape
    margin = int(np.ceil(min_distance_ratio * np.nanmax(rmap)))

    args_list = []
    for x0 in range(0, w - margin, block_size):
        for y0 in range(0, h - margin, block_size):
            r, g, block_args = make_block(y0, x0, rmap, gmap, block_size, margin)
            args_list.append((r, g, block_args, min_distance_ratio))

    out = []
    for args in tqdm(args_list, total=len(args_list), desc='reduce', ncols=150):
        out.append(reduce_peaks_mesh(*args))
    ylist, xlist = [np.concatenate(v, axis=0) for v in zip(*out, strict=False)]
    tlist = tmap[ylist, xlist]
    rlist = rmap[ylist, xlist]
    glist = gmap[ylist, xlist]

    idx = np.flip(np.argsort(glist))
    idx = idx[: np.count_nonzero(glist > 0)]
    return tlist[idx], rlist[idx], ylist[idx], xlist[idx], glist[idx]


def make_block(y0, x0, rsmap, gsmap, block_size, margin):
    h, w = rsmap.shape
    y1, x1 = y0 + block_size, x0 + block_size
    ym, xm = max(y0 - margin, 0), max(x0 - margin, 0)
    yp, xp = min(y1 + margin, h), min(x1 + margin, w)
    block_args = ym, xm, y0, x0, y1, x1
    rmap = rsmap[ym:yp, xm:xp]
    gmap = gsmap[ym:yp, xm:xp]
    return rmap, gmap, block_args


def reduce_peaks_mesh(rmap, gmap, block_args, min_distance_ratio) -> tuple[Array, Array]:
    ylist, xlist = reduce_peaks(rmap, gmap, min_distance_ratio)
    ym, xm, y0, x0, y1, x1 = block_args
    ylist += ym
    xlist += xm
    is_in_block = (y0 <= ylist) & (ylist < y1) & (x0 <= xlist) & (xlist < x1)
    return ylist[is_in_block], xlist[is_in_block]


def reduce_peaks(rmap: Array, gmap: Array, min_distance_ratio: float) -> tuple[Array, Array]:
    h, w = rmap.shape
    ymap, xmap = np.mgrid[:h, :w]

    ys, xs, rs, gs = (np.ravel(a) for a in (ymap, xmap, rmap, gmap))
    n = np.count_nonzero(np.isfinite(gmap))
    ids = np.flip(np.argsort(np.nan_to_num(gs, nan=0)))[:n]

    active_list = []
    while ids.size > 0:
        i, ids = ids[0], ids[1:]
        y0, x0, r0 = ys[i], xs[i], rs[i]
        yc, xc = ys[active_list], xs[active_list]
        dist1 = np.hypot(xc - x0, yc - y0) / r0
        if np.all(dist1 >= min_distance_ratio):
            y1, x1 = ys[ids], xs[ids]
            dist2 = np.hypot(x1 - x0, y1 - y0) / r0
            ids = ids[dist2 >= min_distance_ratio]
            active_list.append(i)

    y, x = ys[active_list], xs[active_list]
    return y, x
