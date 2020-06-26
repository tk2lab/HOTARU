import tensorflow as tf
import numpy as np


def reduce_peak(peaks, radius, thr_distance):
    idx = reduce_peak_idx(peaks, radius, thr_distance)
    return tuple(v[idx] for v in peaks)


def reduce_peak_idx(peaks, radius, thr_distance):
    ts, rs, ys, xs = peaks[:, 0], peaks[:, 1], peaks[:, 2], peaks[:, 3]
    rs = np.array(radius)[rs]
    total = ts.size
    flg = np.arange(total, dtype=np.int32)
    idx = []
    while flg.size > 0:
        i, j = flg[0], flg[1:]
        y0, x0 = ys[i], xs[i]
        y1, x1 = ys[j], xs[j]
        ya, xa = ys[idx], xs[idx]
        thr = np.square(thr_distance * rs[i])
        if not idx or np.all(np.square(ya - y0) + np.square(xa - x0) >= thr):
            cond = np.square(y1 - y0) + np.square(x1 - x0) >= thr
            flg = j[cond]
            idx.append(i)
        else:
            flg = j
    return np.array(idx, np.int32)
