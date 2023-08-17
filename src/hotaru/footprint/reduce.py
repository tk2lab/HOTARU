import multiprocessing as mp
from logging import getLogger

import numpy as np
import pandas as pd

logger = getLogger(__name__)


def reduce_peaks_simple(ys, xs, rs, vs, min_radius, max_radius, min_distance_ratio):
    n = np.count_nonzero(np.isfinite(vs))
    flg = np.argsort(vs)[::-1][:n]
    cell = []
    bg = []
    while flg.size > 0:
        i, flg = flg[0], flg[1:]
        r0 = rs[i]
        if r0 >= min_radius:
            y0, x0 = ys[i], xs[i]
            thr = min_distance_ratio * r0
            if r0 <= max_radius:
                yc, xc = ys[cell], xs[cell]
                if not cell or np.all(np.hypot(xc - x0, yc - y0) >= thr):
                    y1, x1 = ys[flg], xs[flg]
                    cond = np.hypot(x1 - x0, y1 - y0) >= thr
                    flg = flg[cond]
                    cell.append(i)
            else:
                yb, xb = ys[bg], xs[bg]
                if not bg or np.all(np.hypot(xb - x0, yb - y0) >= thr):
                    bg.append(i)
    return cell, bg


def reduce_peaks_mesh(rs, vs, *args, **kwargs):
    h, w = rs.shape
    ys, xs = np.mgrid[:h, :w]
    ys, xs, rs, vs = ys.ravel(), xs.ravel(), rs.ravel(), vs.ravel()
    cell, bg = reduce_peaks_simple(ys, xs, rs, vs, *args, **kwargs)
    return ys[cell], xs[cell], ys[bg], xs[bg]


def reduce_peaks(peakval, density, block_size):
    logger.info("reduce_peaks: %s %d", density, block_size)

    radius, ts, ri, vs = peakval
    rs = radius[ri]
    h, w = rs.shape
    margin = int(np.ceil(density.min_distance_ratio.reduce * rs.max()))

    static_args = (
        density.min_radius,
        density.max_radius,
        density.min_distance_ratio.reduce,
    )

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
            args.append(((y0, x0, ys, xs, ye, xe), (r, v, *static_args)))

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


def _reduce_peaks(args):
    def fix(y, x):
        y += y0
        x += x0
        cond = (ys <= y) & (y < ye) & (xs <= x) & (x < xe)
        return y[cond], x[cond]

    y0, x0, ys, xs, ye, xe = args[0]
    celly, cellx, bgy, bgx = reduce_peaks_mesh(*args[1])
    return *fix(celly, cellx), *fix(bgy, bgx)
