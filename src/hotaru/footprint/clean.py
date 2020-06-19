import tensorflow.keras.backend as K
import tensorflow as tf

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.gaussian import gaussian
from ..image.filter.laplace import gaussian_laplace_multi
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def clean_footprint(data, mask, gauss, radius, batch):
    strategy = tf.distribute.get_strategy()

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)
    to_dense = ToDense(mask)

    mask = K.constant(mask, tf.bool)
    gauss = K.constant(gauss)
    radius = K.constant(radius)

    nk = data.shape[0]
    prog = tf.keras.utils.Progbar(nk)
    ps = tf.TensorArray(tf.float32, nk)
    rs = tf.TensorArray(tf.float32, nk)
    ys = tf.TensorArray(tf.int32, nk)
    i = tf.constant(0)
    for data in strategy.experimental_distribute_dataset(dataset):
        gl, ll, rl, yl, xl = _prepare(data, mask, gauss, radius)
        nx = tf.size(rl)
        for k in tf.range(nx):
            g, l, r, y, x = gl[k], ll[k], rl[k], yl[k], xl[k]
            pos = get_segment_index(l, y, x, mask)
            val = get_normalized_val(g, pos)
            footprint = to_dense(pos, val)
            firmness = get_magnitude(l, pos) / get_magnitude(g, pos)
            ps = ps.write(i, footprint)
            rs = rs.write(i, [r, firmness])
            ys = ys.write(i, [y, x])
            i += 1
            prog.add(1)
    return ps.stack().numpy(), rs.stack().numpy(), ys.stack().numpy()


@distributed(*[ReduceOp.CONCAT for _ in range(5)], loop=False)
def _prepare(imgs, mask, gauss, radius):
    gs = gaussian(imgs, gauss) if gauss > 0.0 else imgs
    ls = gaussian_laplace_multi(gs, radius)
    nk, nr, h, w = tf.shape(ls)[0], tf.shape(ls)[1], tf.shape(ls)[2], tf.shape(ls)[3]
    lsr = K.reshape(ls, (nk, -1))
    pos = tf.cast(K.argmax(lsr, axis=1), tf.int32)
    rs = pos // h // w
    ys = (pos % (h * w)) // w
    xs = (pos % (h * w)) % w
    ls = tf.gather_nd(ls, tf.stack([tf.range(nk), rs], axis=1))
    rs = tf.gather(radius, rs)
    return imgs, ls, rs, ys, xs
