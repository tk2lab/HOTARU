import tensorflow as tf


def median(x):
    m = x.shape[1] // 2
    k = tf.nn.top_k(x, m, sorted=False).values
    return tf.reduce_min(k, axis=1)


def calc_overwrap(x):
    x = tf.cast(x, tf.float32)
    s = tf.math.reduce_sum(x, axis=1)
    c = tf.linalg.matmul(x, x, False, True) / tf.where(s > 0, s, 1)
    c = tf.linalg.set_diag(c, tf.zeros(tf.size(s)))
    c = tf.linalg.band_part(c, 0, -1)
    return tf.math.reduce_max(c, axis=0)


def calc_denseness(x):
    xmed = median(x)
    xmad = median(tf.math.abs(x - xmed[:, None]))
    xmax = tf.math.reduce_max(x, axis=1)
    return xmad / xmax
