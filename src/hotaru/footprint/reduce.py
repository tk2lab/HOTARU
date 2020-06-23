import tensorflow as tf
import numpy as np


def reduce_peak(peaks, thr_dist):
    idx = reduce_peak_idx(peaks, thr_dist)
    return tuple(v[idx] for v in peaks)


def reduce_peak_idx(peaks, thr_dist):
    ts, rs, ys, xs = peaks[:, 0], peaks[:, 1], peaks[:, 2], peaks[:, 3]
    total = ts.size
    prog = tf.keras.utils.Progbar(total)
    flg = np.arange(total, dtype=np.int32)
    idx = []
    while flg.size > 0:
        i, j = flg[0], flg[1:]
        y0, x0 = ys[i], xs[i]
        y1, x1 = ys[j], xs[j]
        thr = np.square(thr_dist * rs[i])
        cond = np.square(y1 - y0) + np.square(x1 - x0) >= thr
        flg = j[cond]
        idx.append(i)
        prog.update(total - flg.size)
    return np.array(idx, np.int32)
