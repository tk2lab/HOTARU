import tensorflow.keras.backend as K
import tensorflow as tf

from ..util.distribute import distributed, ReduceOp


def calc_std(data, mask, nt=None, verbose=1):

    @distributed(ReduceOp.CONCAT, ReduceOp.SUM, ReduceOp.SUM, ReduceOp.SUM)
    def _calc(imgs, mask):
        masked = tf.boolean_mask(imgs, mask, axis=1)
        avg_t = K.mean(masked, axis=1)
        sum_x = K.sum(imgs, axis=0)
        sumsq = K.sum(K.square(masked - avg_t[:, None]))
        nt = K.cast_to_floatx(K.shape(imgs)[0])
        return avg_t, sum_x, sumsq, nt

    prog = tf.keras.utils.Progbar(nt, verbose=verbose)
    mask = K.constant(mask, tf.bool)
    nx = K.cast_to_floatx(tf.math.count_nonzero(mask))
    avg_t, sum_x, sumsq, nt = _calc(data, mask, prog=prog)
    avg_x = sum_x / nt
    avg_0 = K.mean(avg_t)
    avg_t -= avg_0
    avgxm = tf.boolean_mask(avg_x - avg_0, mask)
    std = K.sqrt(sumsq / nt / nx - K.mean(K.square(avgxm)))
    return avg_t.numpy(), avg_x.numpy(), std.numpy()
