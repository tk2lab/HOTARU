import multiprocessing as mp

import numpy as np
import pandas as pd


def reduce_peak(ts, rs, vs, rmin, rmax, thr_distance):
    h, w = vs.shape
    idx = np.argsort(vs.ravel())[::-1]
    ys, xs = np.divmod(idx, w)
    ts = ts[ys, xs]
    rs = rs[ys, xs]
    flg = np.arange(h * w)
    out = []
    while flg.size > 0:
        i, j = flg[0], flg[1:]
        y0, x0 = ys[i], xs[i]
        y1, x1 = ys[j], xs[j]
        yo, xo = ys[out], xs[out]
        thr = thr_distance * rs[i]
        if (rmin < rs[i] < rmax) and (
            not out or np.all(np.hypot(xo - x0, yo - y0) >= thr)
        ):
            cond = np.hypot(x1 - x0, y1 - y0) >= thr
            flg = j[cond]
            out.append(i)
        else:
            flg = j
    return ys[out], xs[out]


def _reduce_peak(args):
    y0, x0, ys, xs, ye, xe = args[0]
    y, x = reduce_peak(*args[1])
    y += y0
    x += x0
    cond = (ys <= y) & (y < ye) & (xs <= x) & (x < xe)
    return y[cond], x[cond]


def reduce_peak_block(peakval, rmin, rmax, thr_distance, block_size):
    ts, rs, vs = peakval
    h, w = vs.shape
    margin = int(np.ceil(thr_distance * rs.max()))
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
            args.append(((y0, x0, ys, xs, ye, xe), (t, r, v, rmin, rmax, thr_distance)))
            print(y0, x0, ys, xs, ye, xe)

    y, x = [], []
    with mp.Pool() as pool:
        for yi, xi in pool.imap_unordered(_reduce_peak, args):
            y.append(yi)
            x.append(xi)

    y = np.concatenate(y, axis=0)
    x = np.concatenate(x, axis=0)
    t = ts[y, x]
    r = rs[y, x]
    v = vs[y, x]
    return (
        pd.DataFrame(dict(t=t, r=r, y=y, x=x, v=v))
        .sort_values("v", ascending=False)
        .reset_index(drop=True)
    )
