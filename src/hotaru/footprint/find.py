import tensorflow.keras.backend as K
import tensorflow as tf
from tqdm import trange

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.gaussian import gaussian
from ..image.filter.laplace import gaussian_laplace_multi


def find_peak(
        data, mask, gauss, radius, thr_intensity, shard,
        batch, nt=None, verbose=1,
):

    @distributed(ReduceOp.STACK, ReduceOp.STACK, ReduceOp.STACK)
    def _find(data, mask, gauss, radius):
        times, imgs = data
        times = tf.cast(times, tf.int32)
        if gauss > 0.0:
            imgs = gaussian(imgs, gauss)
        gl = gaussian_laplace_multi(imgs, radius)
        max_gl = tf.nn.max_pool3d(
            gl[..., None], [1, 3, 3, 3, 1], [1, 1, 1, 1, 1], padding='SAME'
        )[..., 0]
        bit = tf.equal(gl, max_gl) & (gl > thr_intensity) & mask
        glp = tf.where(bit, gl, tf.cast(0, gl.dtype))
        shape = tf.shape(glp)
        glp = tf.reshape(glp, (-1, shape[2], shape[3]))
        pos = tf.math.argmax(glp, output_type=tf.int32)
        t = tf.gather(times, pos // shape[1])
        r = pos % shape[1]
        g = tf.reduce_max(glp, axis=0)
        return t, r, g

    data = unmasked(data, mask)
    data = data.enumerate().shard(shard, 0).batch(batch)
    mask = K.constant(mask, tf.bool)
    shape = K.shape(mask)
    gauss = K.constant(gauss)
    radius = K.constant(radius)
    total = (nt + shard - 1) // shard
    with trange(total, desc='Find', disable=verbose == 0) as prog:
        t, r, g = _find(data, mask, gauss, radius, prog=prog)
    idx = tf.math.argmax(g, axis=0, output_type=tf.int32)
    x, y = tf.meshgrid(tf.range(shape[1]), tf.range(shape[0]))
    idx = tf.reshape(tf.stack((idx, y, x), axis=2), (-1, 3))
    t = tf.gather_nd(t, idx)
    r = tf.gather_nd(r, idx)
    x = tf.reshape(x, -1)
    y = tf.reshape(y, -1)
    g = tf.gather_nd(g, idx)
    idx = tf.argsort(g, direction='DESCENDING')
    cond = tf.gather(g, idx) > 0.0
    idx = tf.boolean_mask(idx, cond)
    t = tf.gather(t, idx)
    r = tf.gather(r, idx)
    x = tf.gather(x, idx)
    y = tf.gather(y, idx)
    g = tf.gather(g, idx)
    return tf.stack((t, r, y, x), axis=1).numpy(), g.numpy()
