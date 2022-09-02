import tensorflow as tf

from ..util.distribute import ReduceOp
from ..util.distribute import distributed


def calc_max(imgs, prog=None):
    @distributed(ReduceOp.SUM)
    def _calc(img):
        img = tf.cast(img, tf.float32)
        return (tf.reduce_sum(img, axis=0),)

    (imax,) = _calc(imgs, prog=prog)
    return imax
