import tensorflow as tf
import numpy as np

from ..util.filter import neighbor


_SUM = tf.distribute.ReduceOp.SUM


def calc_stats(data, mask, nt):

    @tf.function
    def _calc(img, mask):
        masked = tf.boolean_mask(img, mask, axis=1)
        avg_t = tf.reduce_mean(masked, axis=1)
        sum_x = tf.reduce_sum(img, axis=0)
        sumsq = tf.reduce_sum(tf.square(masked - mean[:, None]))
        return avg_t, sum_x, sumsq

    strategy = tf.distribute.get_strategy()
    dist_data = strategy.experimental_distribute_dataset(data)
    dist_mask = strategy.experimental_distribute_values_from_function(lambda: mask)

    avg_t = tf.zeros((nt,), tf.float32)
    sum_x = tf.zeros_like(mask, tf.float32)
    sumsq = tf.constant(0.0)

    e = tf.constant(0)
    for dist_img in dist_data:
        s, e = e, e + tf.shape(img)[0]
        trange = tf.range(s, e)[:, None]
        dist_avg_t, dist_sum_x, dist_sumsq = strategy.run(dist_calc, (dist_img, dist_mask))
        local_avg_t = tf.concat(strategy.experimental_local_results(dist_avg_t), axis=0)
        local_sum_x = strategy.reduce(_SUM, dist_sum_x, 0)
        local_sumsq = strategy.reduce(_SUM, dist_sumsq)
        avg_t = tf.tensor_scatter_nd_update(avg_t, trange, local_avg_t)
        sum_x += local_sum_x
        sumsq += local_sumsq

    nt = tf.cast(nt, tf.float32)
    nx = tf.cast(tf.math.count_nonzero(mask), tf.float32)
    avg_0 = tf.reduce_mean(avg_t)
    avg_t -= avg_0
    avg_x = sum_x / nt
    avgxm = tf.boolean_mask(avg_x - avg0, mask)
    std = tf.sqrt(sumsq / nt / nx - tf.reduce_mean(tf.square(avgxm)))
    return avg_t, avg_x, std


@tf.function
def calc_max(data):
    strategy = tf.distribute.get_strategy()
    dist_data = strategy.experimental_distribute_dataset(data)
    mxx = tf.constant(-np.Inf)
    for dist_img in dist_data:
        dist_max = strategy.run(tf.reduce_max, (dist_img,), dict(axis=0))
        local_max = strategy.experimental_local_results(dist_max)
        mxx = tf.where(mxx > local_max, mxx, local_max)
    return mxx


@tf.function
def calc_cor(imgs):
    sx1, sx2, sy1, sy2, sxy = (tf.constant(0.0) for _ in range(5))
    count = tf.constant(0, tf.int32)
    for img in imgs:
        nei = neighbor(img)
        sx1 += tf.reduce_sum(img, axis=0)
        sy1 += tf.reduce_sum(nei, axis=0)
        sx2 += tf.reduce_sum(tf.square(img), axis=0)
        sxy += tf.reduce_sum(img * nei, axis=0)
        sy2 += tf.reduce_sum(tf.square(nei), axis=0)
        count += tf.shape(img)[0]
    ntf = tf.cast(count, tf.float32)
    avg_x = sx1 / ntf
    avg_y = sy1 / ntf
    cov_xx = sx2 / ntf - tf.square(avg_x)
    cov_xy = sxy / ntf - avg_x * avg_y
    cov_yy = sy2 / ntf - tf.square(avg_y)
    return cov_xy / tf.sqrt(cov_xx * cov_yy)    
