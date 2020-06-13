import tensorflow.keras.backend as K
import tensorflow as tf

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.gaussian import gaussian
from ..image.filter.laplace import gaussian_laplace_multi


def find_peak(data, mask, gauss, radius, thr_gl, batch, prog=None):

    @distributed(*[ReduceOp.CONCAT]*5)
    def _find(data, mask, gauss, radius, thr_gl):
        times, imgs = data
        if gauss > 0.0:
            imgs = gaussian(imgs, gauss)
        gl = gaussian_laplace_multi(imgs, radius)
        max_gl = tf.nn.max_pool3d(
            gl[..., None], [1, 3, 3, 3, 1], [1, 1, 1, 1, 1], padding='SAME'
        )[..., 0]
        bit = tf.equal(gl, max_gl) & (gl > thr_gl) & mask
        posr = tf.cast(tf.where(bit), tf.int32)
        ts = tf.gather(times, posr[:, 0])
        rs = tf.gather(radius, posr[:, 1])
        ys = posr[:, 2]
        xs = posr[:, 3]
        gs = tf.gather_nd(gl, posr)
        return ts, rs, ys, xs, gs

    data = unmasked(data, mask)
    data = data.enumerate().batch(batch)
    mask = K.constant(mask, tf.bool)
    gauss = K.constant(gauss)
    radius = K.constant(radius)
    thr_gl = K.constant(thr_gl)
    peak = _find(data, mask, gauss, radius, thr_gl, prog=prog)
    idx = tf.argsort(peak[-1])[::-1]
    return tuple(tf.gather(v, idx).numpy() for v in peak)
