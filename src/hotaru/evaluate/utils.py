import numpy as np
import tensorflow as tf


def median(x):
    m = x.get_shape()[1] // 2
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


def _calc_area(x, threshold):
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    x = (x - xmin) / (xmax - xmin)
    return np.count_nonzero(x > threshold, axis=1)


def _calc_denseness(x, scale=100):
    def sp(x):
        n, b = np.histogram(x[x>0], bins=np.linspace(0, 1, scale + 1))
        return b[np.argmax(n)]
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    x = (x - xmin) / (xmax - xmin)
    return np.array([sp(xi) for xi in x])
