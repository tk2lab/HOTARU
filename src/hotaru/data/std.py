import tensorflow as tf

from ..util.distribute import distributed, ReduceOp


def calc_std(data, mask, nt, batch):

    @distributed(ReduceOp.CONCAT, ReduceOp.SUM, ReduceOp.SUM)
    def _calc(imgs, mask):
        masked = tf.boolean_mask(imgs, mask, axis=1)
        avg_t = tf.reduce_mean(masked, axis=1)
        sum_x = tf.reduce_sum(imgs, axis=0)
        sumsq = tf.reduce_sum(tf.square(masked - avg_t[:, None]))
        return avg_t, sum_x, sumsq

    prog = tf.keras.utils.Progbar(nt)
    nt = tf.cast(nt, tf.float32)
    nx = tf.cast(tf.math.count_nonzero(mask), tf.float32)
    avg_t, sum_x, sumsq = _calc(data.batch(batch), mask, prog=prog)
    avg_0 = tf.reduce_mean(avg_t)
    avg_t -= avg_0
    avg_x = sum_x / nt
    avgxm = tf.boolean_mask(avg_x - avg_0, mask)
    std = tf.sqrt(sumsq / nt / nx - tf.reduce_mean(tf.square(avgxm)))
    return std, avg_t, avg_x
