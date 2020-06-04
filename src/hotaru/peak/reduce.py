import tensorflow as tf
import numpy as np


def reduce_peak(peaks, thr_dist):
    idx = reduce_peak_idx(peaks, thr_dist)
    return tuple(tf.gather(v, idx) for v in peaks)


def reduce_peak_idx(peaks, thr_dist):
    ts, rs, ys, xs, gs = peaks
    yfs = tf.cast(ys, tf.float32)
    xfs = tf.cast(xs, tf.float32)
    #thr_dist = tf.cast(thr_dist, tf.float32)
    flg = tf.range(tf.size(gs), dtype=tf.int32)
    idx = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    total = tf.size(ts).numpy()
    prog = tf.keras.utils.Progbar(total)
    while tf.size(flg) > 0:
        i, j = flg[0], flg[1:]
        y0, x0 = yfs[i], xfs[i]
        y1, x1 = tf.gather(yfs, j), tf.gather(xfs, j)
        thr = tf.square(thr_dist * rs[i])
        cond = tf.square(y1 - y0) + tf.square(x1 - x0) >= thr
        flg = tf.boolean_mask(j, cond)
        idx = idx.write(idx.size(), i)
        prog.update((total - tf.size(flg)).numpy())
    return idx.stack()
