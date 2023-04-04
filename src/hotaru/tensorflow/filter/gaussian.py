import math

import tensorflow as tf


def gaussian(imgs, r):
    sqrt_2pi = tf.constant(math.sqrt(2 * math.pi))
    mr = tf.math.ceil(r)
    d = tf.math.square(tf.range(-4 * mr, 4 * mr + 1, 1))
    r2 = tf.math.square(r)
    o0 = tf.math.exp(-d / r2 / 2) / r / sqrt_2pi
    tmp = imgs[..., None]
    tmp = tf.nn.conv2d(tmp, tf.reshape(o0, (1, -1, 1, 1)), (1, 1), "SAME")
    tmp = tf.nn.conv2d(tmp, tf.reshape(o0, (-1, 1, 1, 1)), (1, 1), "SAME")
    return tmp[..., 0]
