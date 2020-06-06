import time

import tensorflow as tf

from ..data.dataset import unmasked
from ..image.gaussian import gaussian
from ..image.laplace import gaussian_laplace_multi
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def make_footprint(dataset, mask, gauss, radius, ts, rs, ys, xs, batch):

    def _select(start, end, ts, rs, ys, xs):
        cond = (start <= ts) & (ts < end)
        ids = tf.cast(tf.where(cond)[:, 0], tf.int32)
        tl = tf.gather(ts, ids) - start
        rl = tf.gather(rs, ids)
        yl = tf.gather(ys, ids)
        xl = tf.gather(xs, ids)
        return tl, rl, yl, xl

    strategy = tf.distribute.get_strategy()

    dataset = unmasked(dataset, mask).batch(batch)
    mask = tf.convert_to_tensor(mask, tf.float32)
    to_dense = ToDense(mask)
    gauss = tf.convert_to_tensor(gauss, tf.float32)
    radius = tf.convert_to_tensor(radius, tf.float32)
    ts = tf.convert_to_tensor(ts, tf.int32)
    rs = tf.convert_to_tensor(rs, tf.int32)
    ys = tf.convert_to_tensor(ys, tf.int32)
    xs = tf.convert_to_tensor(xs, tf.int32)

    prog = tf.keras.utils.Progbar(tf.size(ts).numpy())
    fps = tf.TensorArray(tf.float32, 0, True)
    mgs = tf.TensorArray(tf.float32, 0, True)
    e = tf.constant(0)
    for imgs in dataset:
        s, e = e, e + tf.shape(imgs)[0]
        tl, rl, yl, xl = _select(s, e, ts, rs, ys, xs)
        if gauss > 0.0:
            img = gaussian(imgs, gauss)
        gls = gaussian_laplace_multi(imgs, radius)

        nk = tf.size(tl)
        for k in tf.range(nk):
            t, r, y, x = tl[k], rl[k], yl[k], xl[k]
            gl = gls[t, :, :, r]
            pos = get_segment_index(gl, y, x, mask)
            val = get_normalized_val(gl, pos)
            mag = get_magnitude(gl, pos)
            footprint = to_dense(pos, val)
            mgs = mgs.write(mgs.size(), mag)
            fps = fps.write(fps.size(), footprint)
            prog.add(1)
    return fps.stack(), mgs.stack()
