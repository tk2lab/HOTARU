import tensorflow as tf
import numpy as np


def calc_distance(peaks, radius):
    rs, ys, xs = peaks[:, 1], peaks[:, 2], peaks[:, 3]
    rs = np.array(radius)[rs]
    total = rs.size
    dist = np.full(total, np.inf, dtype=np.float32)
    for i in range(1, total):
        rl, xl, yl = rs[:i], xs[:i], ys[:i]
        r, x, y = rs[i], xs[i], ys[i]
        rl = np.where(rl > r, rl, r)
        dist[i] = np.min(np.sqrt((x - xl)**2 + (y - yl)**2) / rl)
    return dist


def reduce_peak(peaks, radius, thr_distance):
    idx = reduce_peak_idx(peaks, radius, thr_distance)
    return tuple(v[idx] for v in peaks)


def reduce_peak_idx(peaks, radius, thr_distance):
    rs, ys, xs = peaks[:, 1], peaks[:, 2], peaks[:, 3]
    rs = np.array(radius)[rs]
    total = rs.size
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
