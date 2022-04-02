import tensorflow as tf
import click

from ..util.distribute import distributed, ReduceOp


def calc_stats(data, mask, nt=None, verbose=1):

    @distributed(ReduceOp.MIN, ReduceOp.MAX, ReduceOp.CONCAT, ReduceOp.SUM, ReduceOp.SUM, ReduceOp.SUM)
    def _calc(imgs, mask):
        masked = tf.boolean_mask(imgs, mask, axis=1)
        min_tx = tf.math.reduce_min(masked)
        max_tx = tf.math.reduce_max(masked)
        avg_t = tf.math.reduce_mean(masked, axis=1)
        sum_x = tf.math.reduce_sum(imgs, axis=0)
        sumsq = tf.math.reduce_sum(tf.math.square(masked - avg_t[:, None]))
        nt = tf.cast(tf.shape(imgs)[0], tf.float32)
        return min_tx, max_tx, avg_t, sum_x, sumsq, nt

    mask = tf.convert_to_tensor(mask, tf.bool)
    nx = tf.cast(tf.math.count_nonzero(mask), tf.float32)
    with click.progressbar(length=nt, label='Calc Stats') as prog:
        min_t, max_t, avg_t, sum_x, sumsq, nt = _calc(data, mask, prog=prog)
    avg_x = sum_x / nt
    avg_0 = tf.math.reduce_mean(avg_t)
    avg_x -= avg_0
    avgxm = tf.boolean_mask(avg_x, mask)
    std = tf.math.sqrt(sumsq / nt / nx - tf.math.reduce_mean(tf.math.square(avgxm)))
    return min_t.numpy(), max_t.numpy(), std.numpy(), avg_t.numpy(), avg_x.numpy()
