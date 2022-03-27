import tensorflow.keras.backend as K
import tensorflow as tf
from tqdm import trange

from ..util.distribute import distributed, ReduceOp


def calc_stats(data, mask, nt=None, verbose=1):

    @distributed(ReduceOp.MIN, ReduceOp.MAX, ReduceOp.CONCAT, ReduceOp.SUM, ReduceOp.SUM, ReduceOp.SUM)
    def _calc(imgs, mask):
        masked = tf.boolean_mask(imgs, mask, axis=1)
        min_tx = K.min(masked)
        max_tx = K.max(masked)
        avg_t = K.mean(masked, axis=1)
        sum_x = K.sum(imgs, axis=0)
        sumsq = K.sum(K.square(masked - avg_t[:, None]))
        nt = K.cast_to_floatx(K.shape(imgs)[0])
        return min_tx, max_tx, avg_t, sum_x, sumsq, nt

    mask = K.constant(mask, tf.bool)
    nx = K.cast_to_floatx(tf.math.count_nonzero(mask))
    with trange(nt, desc='Calc Stats', disable=verbose == 0) as prog:
        min_t, max_t, avg_t, sum_x, sumsq, nt = _calc(data, mask, prog=prog)
    avg_x = sum_x / nt
    avg_0 = K.mean(avg_t)
    avg_x -= avg_0
    avgxm = tf.boolean_mask(avg_x, mask)
    std = K.sqrt(sumsq / nt / nx - K.mean(K.square(avgxm)))
    return min_t.numpy(), max_t.numpy(), std.numpy(), avg_t.numpy(), avg_x.numpy()
