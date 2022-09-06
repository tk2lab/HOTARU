import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed


def calc_std(imgs, prog=None):
    @distributed(ReduceOp.SUM, ReduceOp.SUM)
    def _calc(img):
        img = tf.cast(img, tf.float32)
        d = img - tf.reduce_mean(img, axis=0)
        s = tf.reduce_sum(d**2, axis=0)
        n = tf.cast(tf.shape(img)[0], tf.float32)
        return s, n

    s, n = _calc(imgs, prog=prog)
    return tf.math.sqrt(s / n)