import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from .neighbor import neighbor


def calc_cor(imgs, prog=None):
    @distributed(*[ReduceOp.SUM for _ in range(6)])
    def _calc(img):
        img = tf.cast(img, tf.float32)
        nei = neighbor(img)
        sx1 = tf.math.reduce_sum(img, axis=0)
        sy1 = tf.math.reduce_sum(nei, axis=0)
        sx2 = tf.math.reduce_sum(tf.math.square(img), axis=0)
        sxy = tf.math.reduce_sum(img * nei, axis=0)
        sy2 = tf.math.reduce_sum(tf.math.square(nei), axis=0)
        ntf = tf.cast(tf.shape(img)[0], tf.float32)
        return sx1, sy1, sx2, sxy, sy2, ntf

    sx1, sy1, sx2, sxy, sy2, ntf = _calc(imgs, prog=prog)
    avg_x = sx1 / ntf
    avg_y = sy1 / ntf
    cov_xx = sx2 / ntf - tf.math.square(avg_x)
    cov_xy = sxy / ntf - avg_x * avg_y
    cov_yy = sy2 / ntf - tf.math.square(avg_y)
    return cov_xy / tf.math.sqrt(cov_xx * cov_yy)
