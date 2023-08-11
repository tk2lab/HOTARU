import multiprocessing as mp
from logging import getLogger

import numpy as np
import pandas as pd

logger = getLogger(__name__)


def reduce_peaks(peakval, min_distance_ratio, block_size, **args):
    radius, ts, rs, vs = peakval
    h, w = vs.shape
    margin = int(np.ceil(min_distance_ratio * rs.max()))
    logger.info(f"reduce_peaks: {radius} {min_distance_ratio} {block_size}")
    args = []
    for xs in range(0, w - margin, block_size):
        x0 = max(xs - margin, 0)
        xe = xs + block_size
        x1 = min(xe + margin, w)
        for ys in range(0, h - margin, block_size):
            y0 = max(ys - margin, 0)
            ye = ys + block_size
            y1 = min(ye + margin, h)
            t = ts[y0:y1, x0:x1]
            r = rs[y0:y1, x0:x1]
            v = vs[y0:y1, x0:x1]
            args.append(
                (
                    (y0, x0, ys, xs, ye, xe),
                    (t, r, v, radius, min_distance_ratio),
                )
            )

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
                radius=radius[rs[y, x]],
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
    return peaks


def reduce_peaks_simple(ts, rs, vs, radius, min_distance_ratio):
    h, w = vs.shape
    n = np.count_nonzero(np.isfinite(vs))
    nr = len(radius)
    idx = np.argsort(vs.ravel())[::-1][:n]
    ys, xs = np.divmod(idx, w)
    ts = ts[ys, xs]
    rs = rs[ys, xs]
    flg = np.arange(idx.size)
    cell = []
    bg = []
    while flg.size > 0:
        i, j = flg[0], flg[1:]
        y0, x0 = ys[i], xs[i]
        y1, x1 = ys[j], xs[j]
        yo, xo = ys[cell], xs[cell]
        thr = min_distance_ratio * radius[rs[i]]
        flg = j
        if rs[i] == 0:
            pass
        elif rs[i] == nr - 1:
            bg.append(i)
        elif not cell or np.all(np.hypot(xo - x0, yo - y0) >= thr):
            cond = np.hypot(x1 - x0, y1 - y0) >= thr
            flg = flg[cond]
            cell.append(i)
    return ys[cell], xs[cell], ys[bg], xs[bg]


def _reduce_peaks(args):
    def fix(y, x):
        y += y0
        x += x0
        cond = (ys <= y) & (y < ye) & (xs <= x) & (x < xe)
        return y[cond], x[cond]

    y0, x0, ys, xs, ye, xe = args[0]
    celly, cellx, bgy, bgx = reduce_peaks_simple(*args[1])
    return *fix(celly, cellx), *fix(bgy, bgx)
