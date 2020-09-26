import tensorflow.keras.backend as K
import tensorflow as tf

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.gaussian import gaussian
from ..image.filter.laplace import gaussian_laplace_multi
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def make_segment(data, mask, gauss, radius, peaks, batch, verbose):
    strategy = tf.distribute.get_strategy()

    ts, rs, ys, xs = peaks[:, 0], peaks[:, 1], peaks[:, 2], peaks[:, 3]
    nk = ts.size

    mask = K.constant(mask, tf.bool)
    to_dense = ToDense(mask)

    gauss = K.constant(gauss)
    radius = K.constant(radius)

    ts = K.constant(ts, tf.int32)
    rs = K.constant(rs, tf.int32)
    ys = K.constant(ys, tf.int32)
    xs = K.constant(xs, tf.int32)

    nx = tf.math.count_nonzero(mask)
    data = data.batch(batch)

    verbose=0
    prog = tf.keras.utils.Progbar(nk, verbose=verbose)
    fps = tf.TensorArray(tf.float32, 0, True)
    e = K.constant(0, tf.int32)
    for imgs in strategy.experimental_distribute_dataset(data):
        s, e = e, e + tf.shape(imgs)[0]
        gl, yl, xl = _prepare(
            imgs, s, e, ts, rs, ys, xs, gauss, radius,
        )
        nk = tf.size(yl)
        for k in tf.range(nk):
            g, y, x = gl[k], yl[k], xl[k]
            pos = get_segment_index(g, y, x, mask)
            val = get_normalized_val(g, pos)
            if get_magnitude(g, pos) > 0.0:
                footprint = to_dense(pos, val)
                fps = fps.write(fps.size(), footprint)
            prog.add(1)
    stack = fps.stack()
    return stack.numpy()


@distributed(*[ReduceOp.CONCAT for _ in range(5)], loop=False)
def _prepare(imgs, start, end, ts, rs, ys, xs, gauss, radius):
    cond = (start <= ts) & (ts < end)
    ids = tf.cast(tf.where(cond)[:, 0], tf.int32)
    tl = tf.gather(ts, ids) - start
    rl = tf.gather(rs, ids)
    yl = tf.gather(ys, ids)
    xl = tf.gather(xs, ids)
    if gauss > 0.0:
        imgs = gaussian(imgs, gauss)
    gls = gaussian_laplace_multi(imgs, radius)
    gl = tf.gather_nd(gls, tf.stack([tl, rl], 1))
    return gl, yl, xl
