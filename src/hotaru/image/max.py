import tensorflow as tf
from tqdm import trange

from ..util.distribute import ReduceOp
from ..util.distribute import distributed


def calc_max(imgs, nt=None, verbose=1):

    @distributed(ReduceOp.SUM)
    def _calc(img):
        img = tf.cast(img, tf.float32)
        return tf.reduce_sum(img, axis=0),

    with trange(nt, desc='Calc Max', disable=verbose == 0) as prog:
        imax, = _calc(imgs, prog=prog)
    return imax
