import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tqdm import trange

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.laplace import gaussian_laplace_multi
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def make_segment(dataset, mask, avgx, peaks, batch, verbose):
    avgx = tf.convert_to_tensor(avgx, tf.float32)
    nk = peaks.shape[0]
    ts = tf.convert_to_tensor(peaks['t'].values, tf.int32)
    xs = tf.convert_to_tensor(peaks['x'].values, tf.int32)
    ys = tf.convert_to_tensor(peaks['y'].values, tf.int32)
    rs = tf.convert_to_tensor(peaks['radius'].values, tf.float32)
    radius, rs = tf.unique(rs)
    idx = tf.argsort(radius)
    radius = tf.gather(radius, idx)
    rs = tf.gather(rs, idx)

    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)
    to_dense = ToDense(mask)

    fps = []
    e = K.constant(0, tf.int32)
    with trange(nk, desc='Make', disable=verbose==0) as prog:
        for imgs in dataset:
            s, e = e, e + tf.shape(imgs)[0]
            gl, yl, xl = _prepare(
                imgs + avgx, s, e, ts, rs, ys, xs, radius,
            )
            mk = tf.size(yl)
            for k in tf.range(mk):
                g, y, x = gl[k], yl[k], xl[k]
                pos = get_segment_index(g, y, x, mask)
                val = get_normalized_val(g, pos)
                if get_magnitude(g, pos) > 0.0:
                    footprint = to_dense(pos, val)
                    fps.append(footprint.numpy())
                prog.update(1)
    return np.array(fps)


@distributed(*[ReduceOp.CONCAT for _ in range(5)], loop=False)
def _prepare(imgs, start, end, ts, rs, ys, xs, radius):
    cond = (start <= ts) & (ts < end)
    ids = tf.cast(tf.where(cond)[:, 0], tf.int32)
    tl = tf.gather(ts, ids) - start
    rl = tf.gather(rs, ids)
    yl = tf.gather(ys, ids)
    xl = tf.gather(xs, ids)
    gls = gaussian_laplace_multi(imgs, radius)
    gl = tf.gather_nd(gls, tf.stack([tl, rl], 1))
    return gl, yl, xl
