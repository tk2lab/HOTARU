import numpy as np
import tensorflow as tf

from ..filter.laplace import gaussian_laplace_multi
from ..util.dataset import unmasked
from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .segment import get_segment_index
from .segment import remove_noise


def make_segment(dataset, mask, peaks, batch, prog=None):
    @distributed(ReduceOp.CONCAT, ReduceOp.CONCAT)
    def _make(data, mask, ts, rs, ys, xs, radius):
        h, w = tf.shape(mask)[0], tf.shape(mask)[1]
        idx, imgs = data
        idx = tf.cast(idx, tf.int32)
        cond = (idx[0] <= ts) & (ts <= idx[-1])
        ids = tf.cast(tf.where(cond)[:, 0], tf.int32)
        tl = tf.gather(ts, ids) - idx[0]
        rl = tf.gather(rs, ids)
        yl = tf.gather(ys, ids)
        xl = tf.gather(xs, ids)
        gls = gaussian_laplace_multi(imgs, radius)
        gl = tf.gather_nd(gls, tf.stack([tl, rl], 1))

        out = tf.TensorArray(tf.float32, size=tf.size(ids), element_shape=[nx])
        for k in tf.range(tf.size(ids)):
            g, y, x = gl[k], yl[k], xl[k]
            pos = get_segment_index(g, y, x, mask)
            val = tf.gather_nd(g, pos)
            gmin = tf.math.reduce_min(val)
            gmax = tf.math.reduce_max(val)
            val = (val - gmin) / (gmax - gmin)
            val = remove_noise(val, scale=100)
            img = tf.scatter_nd(pos, val, [h, w])
            img = tf.boolean_mask(img, mask)
            out = out.write(k, img)
        return ids, out.stack()

    nk = peaks.shape[0]
    nx = mask.sum()
    ts = tf.convert_to_tensor(peaks["t"].values, tf.int32)
    xs = tf.convert_to_tensor(peaks["x"].values, tf.int32)
    ys = tf.convert_to_tensor(peaks["y"].values, tf.int32)
    rs = tf.convert_to_tensor(peaks["radius"].values, tf.float32)
    radius, rs = tf.unique(rs)

    dataset = unmasked(dataset, mask)
    dataset = dataset.enumerate().batch(batch)
    ids, segment = _make(dataset, mask, ts, rs, ys, xs, radius, prog=prog)

    segment = segment.numpy()
    out = np.empty_like(segment)
    for i, j in enumerate(ids.numpy()):
        out[j] = segment[i]
    return out
