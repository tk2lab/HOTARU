import multiprocessing as mp
from logging import getLogger

import numpy as np

from ..saving import Data
from ..typing import Array
from .radius import Radius
from .reduce_driver import reduce_peaks_simple

logger = getLogger(__name__)


class PeakList(Data):
    radius: Radius
    ts: Array
    ri: Array
    ys: Array
    xs: Array
    gs: Array


def reduce_peak(
    self,
    min_radius: float,
    max_radius: float,
    min_distance_ratio: float,
    block_size: int,
) -> tuple[PeakList, PeakList]:
    radius, ts, ri, vs = self.radius, self.ts, self.rs, self.gs
    rs = radius[ri]
    h, w = rs.shape
    margin = int(np.ceil(min_distance_ratio * rs.max()))
    reduce_args = min_radius, max_radius, min_distance_ratio

    args = []
    for xs in range(0, w - margin, block_size):
        for ys in range(0, h - margin, block_size):
            r, v, block_args = make_block(ys, xs, rs, vs, h, w, block_size, margin)
            args.append((r, v, reduce_args, block_args))

    out = []
    with mp.Pool() as pool:
        for o in pool.imap_unordered(reduce_peaks_mesh, args):
            out.append(o)
    cy, cx, by, bx = [np.concatenate(v, axis=0) for v in zip(*out, strict=False)]

    cell = PeakList(self.radius, ts[cy, cx], cy, cx, ri[cy, cx], vs[cy, cx])
    bact = PeakList(self.radius, ts[by, bx], by, bx, ri[by, bx], vs[by, bx])
    return cell, bact


def make_block(ys, xs, rs, vs, h, w, block_size, margin):
    x0 = max(xs - margin, 0)
    xe = xs + block_size
    x1 = min(xe + margin, w)
    y0 = max(ys - margin, 0)
    ye = ys + block_size
    y1 = min(ye + margin, h)
    r = rs[y0:y1, x0:x1]
    v = vs[y0:y1, x0:x1]
    block_args = y0, x0, ys, xs, ye, xe
    return r, v, block_args


def reduce_peaks_mesh(args) -> tuple[Array, Array, Array, Array]:
    r, v, reduce_args, block_args = args
    h, w = r.shape
    y, x = np.mgrid[:h, :w]
    cell, bg, _remove = reduce_peaks_simple(y, x, r, v, *reduce_args)
    celly, cellx, bgy, bgx = y[cell], x[cell], y[bg], x[bg]
    cy, cx = select_in_block(celly, cellx, *block_args)
    by, bx = select_in_block(bgy, bgx, *block_args)
    return cy, cx, by, bx


def select_in_block(y, x, y0, x0, ys, xs, ye, xe) -> tuple[Array, Array]:
    y += y0
    x += x0
    cond = (ys <= y) & (y < ye) & (xs <= x) & (x < xe)
    return y[cond], x[cond]
