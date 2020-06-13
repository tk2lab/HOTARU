import tensorflow as tf

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.gaussian import gaussian
from ..image.filter.laplace import gaussian_laplace_multi
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def clean_footprint(data, mask, gauss, radius, batch):
    strategy = tf.distribute.get_strategy()

    dataset = tf.data.Dataset.from_generator(_gen(data), tf.float32)
    mask = tf.convert_to_tensor(mask, tf.bool)
    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)
    to_dense = ToDense(mask)

    gauss = tf.convert_to_tensor(gauss)
    radius = tf.convert_to_tensor(radius)

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


def _gen(data):
    def func():
        for d in data:
            yield tf.convert_to_tensor(d, tf.float32)
    return func


@distributed(*[ReduceOp.CONCAT for _ in range(5)], loop=False)
def _prepare(imgs, mask, gauss, radius):
    gs = gaussian(imgs, gauss) if gauss > 0.0 else imgs
    ls = gaussian_laplace_multi(gs, radius)
    max_gls = tf.reduce_max(ls, axis=(1, 2, 3), keepdims=True)
    pos_bin = tf.equal(ls, max_gls) & mask
    pos = tf.cast(tf.where(pos_bin), tf.int32)
    ls = tf.gather_nd(ls, pos[:, :2])
    rs = tf.gather(radius, pos[:, 1])
    return imgs, ls, rs, pos[:, 2], pos[:, 3]
