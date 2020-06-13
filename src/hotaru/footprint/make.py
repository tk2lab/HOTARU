import tensorflow as tf

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.gaussian import gaussian
from ..image.filter.laplace import gaussian_laplace_multi
from .segment import get_segment_index
from .util import get_normalized_val, get_magnitude, ToDense


def make_footprint(dataset, mask, gauss, radius, peaks, batch):
    strategy = tf.distribute.get_strategy()

    mask = tf.convert_to_tensor(mask, tf.bool)
    dataset = unmasked(dataset, mask)
    dataset = dataset.batch(batch)
    to_dense = ToDense(mask)

    gauss = tf.convert_to_tensor(gauss, tf.float32)
    radius = tf.convert_to_tensor(radius, tf.float32)

    ts, rs, ys, xs = peaks[:4]
    ts = tf.convert_to_tensor(ts, tf.int32)
    rs = tf.convert_to_tensor(rs, tf.int32)
    ys = tf.convert_to_tensor(ys, tf.int32)
    xs = tf.convert_to_tensor(xs, tf.int32)

    prog = tf.keras.utils.Progbar(tf.size(ts).numpy())
    fps = tf.TensorArray(tf.float32, 0, True)
    e = tf.constant(0)
    for imgs in strategy.experimental_distribute_dataset(dataset):
        s, e = e, e + tf.shape(imgs)[0]
        gs, yl, xl = _prepare(
            imgs, s, e, ts, rs, ys, xs, gauss, radius,
        )
        nk = tf.size(yl)
        for k in tf.range(nk):
            g, y, x = gl[k], yl[k], xl[k]
            pos = get_segment_index(gl, y, x, mask)
            val = get_normalized_val(gl, pos)
            footprint = to_dense(pos, val)
            fps = fps.write(fps.size(), footprint)
            prog.add(1)
    return fps.stack().numpy()


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
