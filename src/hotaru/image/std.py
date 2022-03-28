import tensorflow as tf
from tqdm import trange

from ..util.distribute import ReduceOp
from ..util.distribute import distributed


def calc_std(imgs, nt=None, verbose=1):

    @distributed(ReduceOp.SUM, ReduceOp.SUM)
    def _calc(img):
        img = tf.cast(img, tf.float32)
        d = img - tf.reduce_mean(img, axis=0)
        s = tf.reduce_sum(d ** 2, axis=0)
        n = tf.cast(tf.shape(img)[0], tf.float32)
        return s, n

    with trange(nt, desc='Calc Std', disable=verbose == 0) as prog:
        s, n = _calc(imgs, prog=prog)
    return tf.math.sqrt(s / n)
