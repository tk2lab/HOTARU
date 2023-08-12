import multiprocessing as mp
from logging import getLogger

import numpy as np
import pandas as pd

logger = getLogger(__name__)


def reduce_peaks(peakval, density, block_size, **args):
    logger.info("reduce_peaks: %s %d", density, block_size)
    radius, ts, ri, vs = peakval
    rs = radius[ri]
    h, w = rs.shape
    margin = int(np.ceil(density.min_distance_ratio * rs.max()))

    args = []
    for xs in range(0, w - margin, block_size):
        x0 = max(xs - margin, 0)
        xe = xs + block_size
        x1 = min(xe + margin, w)
        for ys in range(0, h - margin, block_size):
            y0 = max(ys - margin, 0)
            ye = ys + block_size
            y1 = min(ye + margin, h)
            r = rs[y0:y1, x0:x1]
            v = vs[y0:y1, x0:x1]
            args.append(((y0, x0, ys, xs, ye, xe), (r, v, density)))

    out = []
    with mp.Pool() as pool:
        for o in pool.imap_unordered(_reduce_peaks, args):
            out.append(o)
    celly, cellx, bgy, bgx = [np.concatenate(v, axis=0) for v in zip(*out)]

    def make_dataframe(y, x):
        df = pd.DataFrame(
            dict(
                y=y,
                x=x,
                t=ts[y, x],
                radius=rs[y, x],
                intensity=vs[y, x],
            )
        )
        return df.sort_values("intensity", ascending=False)

    cell = make_dataframe(celly, cellx)
    cell["kind"] = "cell"
    bg = make_dataframe(bgy, bgx)
    bg["kind"] = "background"
    peaks = pd.concat([cell, bg], axis=0)
    peaks = peaks.reset_index(drop=True)
    peaks.insert(0, "uid", peaks.index)
    for r in radius:
        logger.info("%f %d", r, (peaks.radius == r).sum())
    return peaks


def reduce_peaks_simple(ys, xs, rs, vs, density, **args):
    min_radius = density.min_radius
    max_radius = density.max_radius
    min_distance_ratio = density.min_distance_ratio

    n = np.count_nonzero(np.isfinite(vs))
    idx = np.argsort(vs)[::-1][:n]
    flg = np.arange(idx.size)
    cell = []
    bg = []
    while flg.size > 0:
        i, j = flg[0], flg[1:]
        y0, x0, r0 = ys[i], xs[i], rs[i]
        y1, x1 = ys[j], xs[j]
        yo, xo = ys[cell], xs[cell]
        thr = min_distance_ratio * r0
        flg = j
        if r0 >= min_radius:
            if not cell or np.all(np.hypot(xo - x0, yo - y0) >= thr):
                cond = np.hypot(x1 - x0, y1 - y0) >= thr
                flg = flg[cond]
                if r0 <= max_radius:
                    cell.append(i)
                else:
                    bg.append(i)
    return cell, bg


def reduce_peaks_mesh(rs, vs, *args, **kwargs):
    h, w = rs.shape
    ys, xs = np.mgrid[:h, :w]
    ys, xs, rs, vs = ys.ravel(), xs.ravel(), rs.ravel(), vs.ravel()
    cell, bg = reduce_peaks_simple(ys, xs, rs, vs, *args, **kwargs)
    return ys[cell], xs[cell], ys[bg], xs[bg]


def _reduce_peaks(args):
    def fix(y, x):
        y += y0
        x += x0
        cond = (ys <= y) & (y < ye) & (xs <= x) & (x < xe)
        return y[cond], x[cond]

    y0, x0, ys, xs, ye, xe = args[0]
    celly, cellx, bgy, bgx = reduce_peaks_mesh(*args[1])
    return *fix(celly, cellx), *fix(bgy, bgx)
