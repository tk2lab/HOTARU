import tensorflow as tf
import numpy as np


def reduce_peak(peaks, thr_dist):
    idx = reduce_peak_idx(peaks, thr_dist)
    return tuple(v[idx] for v in peaks)


def reduce_peak_idx(peaks, thr_dist):
    ts, rs, ys, xs, gs = peaks
    ts = tf.convert_to_tensor(ts, tf.int32)
    rs = tf.convert_to_tensor(rs, tf.float32)
    ys = tf.convert_to_tensor(ys, tf.float32)
    xs = tf.convert_to_tensor(xs, tf.float32)
    thr_dist = tf.convert_to_tensor(thr_dist, tf.float32)
    flg = tf.range(tf.size(gs), dtype=tf.int32)
    idx = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    total = tf.size(ts).numpy()
    prog = tf.keras.utils.Progbar(total)
    while tf.size(flg) > 0:
        i, j = flg[0], flg[1:]
        y0, x0 = ys[i], xs[i]
        y1, x1 = tf.gather(ys, j), tf.gather(xs, j)
        thr = tf.square(thr_dist * rs[i])
        cond = tf.square(y1 - y0) + tf.square(x1 - x0) >= thr
        flg = tf.boolean_mask(j, cond)
        idx = idx.write(idx.size(), i)
        prog.update((total - tf.size(flg)).numpy())
    return idx.stack().numpy()
