import tensorflow.keras.backend as K
import tensorflow as tf

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.gaussian import gaussian
from ..image.filter.laplace import gaussian_laplace_multi


def find_peak(
        data, mask, gauss, radius, thr_intensity, shard,
        batch, nt=None, verbose=1,
):

    @distributed(ReduceOp.CONCAT, ReduceOp.CONCAT)
    def _find(data, mask, gauss, radius, thr_intensity):
        times, imgs = data
        times = tf.cast(times, tf.int32)
        if gauss > 0.0:
            imgs = gaussian(imgs, gauss)
        gl = gaussian_laplace_multi(imgs, radius)
        max_gl = tf.nn.max_pool3d(
            gl[..., None], [1, 3, 3, 3, 1], [1, 1, 1, 1, 1], padding='SAME'
        )[..., 0]
        bit = tf.equal(gl, max_gl) & (gl > thr_intensity) & mask
        posr = tf.cast(tf.where(bit), tf.int32)
        ts = tf.gather(times, posr[:, 0])
        rs = posr[:, 1]
        ys = posr[:, 2]
        xs = posr[:, 3]
        gs = tf.gather_nd(gl, posr)
        return tf.stack((ts, rs, ys, xs), axis=1), gs

    data = unmasked(data, mask)
    data = data.enumerate().shard(shard, 0).batch(batch)
    mask = K.constant(mask, tf.bool)
    gauss = K.constant(gauss)
    radius = K.constant(radius)
    thr_intensity = K.constant(thr_intensity)
    prog = tf.keras.utils.Progbar((nt + shard - 1) // shard, verbose=verbose)
    pos, score = _find(data, mask, gauss, radius, thr_intensity, prog=prog)
    idx = tf.argsort(score)[::-1]
    return tf.gather(pos, idx).numpy(), tf.gather(score, idx).numpy()
