import tensorflow as tf
import numpy as np

from ..util.distribute import distributed, ReduceOp
from ..image.gaussian import gaussian
from ..image.laplace import gaussian_laplace_multi


def find_peak(data, mask, gauss, radius, thr_gl, batch, nt=None):

    @distributed(*[ReduceOp.CONCAT]*5)
    def _find(data, mask, gauss, radius, thr_gl):
        times, imgs = data
        if gauss > 0.0:
            imgs = gaussian(imgs, gauss)
        gl = gaussian_laplace_multi(imgs, radius)
        max_gl = tf.nn.max_pool3d(
            gl[..., None], [1, 3, 3, 3, 1], [1, 1, 1, 1, 1], padding='SAME'
        )[..., 0]
        bit = tf.equal(gl, max_gl) & (gl > thr_gl) & mask[..., None]
        posr = tf.cast(tf.where(bit), tf.int32)
        ts = tf.gather(times, posr[:, 0])
        ys = posr[:, 1]
        xs = posr[:, 2]
        rs = tf.gather(radius, posr[:, 3])
        gs = tf.gather_nd(gl, posr)
        return ts, rs, ys, xs, gs

    data = data.enumerate().batch(batch)
    radius = tf.convert_to_tensor(radius)
    prog = tf.keras.utils.Progbar(nt)
    peak = _find(data, mask, gauss, radius, thr_gl, prog=prog)
    idx = tf.argsort(peak[-1])[::-1]
    return tuple(tf.gather(v, idx) for v in peak)
