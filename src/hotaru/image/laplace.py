import math

import tensorflow as tf


def gaussian_laplace(imgs, r):
    sqrt_2pi = tf.constant(math.sqrt(2 * math.pi))
    mr = tf.math.ceil(r)
    d = tf.math.square(tf.range(-4 * mr, 4 * mr + 1, 1))
    r2 = tf.math.square(r)
    o0 = tf.math.exp(-d / r2 / 2) / r / sqrt_2pi
    o2 = (1 - d / r2) * o0
    tmp = imgs[..., None]
    gl1 = tf.nn.conv2d(tmp, tf.reshape(o2, (1, -1, 1, 1)), (1, 1), "SAME")
    gl1 = tf.nn.conv2d(gl1, tf.reshape(o0, (-1, 1, 1, 1)), (1, 1), "SAME")
    gl2 = tf.nn.conv2d(tmp, tf.reshape(o2, (-1, 1, 1, 1)), (1, 1), "SAME")
    gl2 = tf.nn.conv2d(gl2, tf.reshape(o0, (1, -1, 1, 1)), (1, 1), "SAME")
    return (gl1 + gl2)[..., 0]


def gaussian_laplace_multi(imgs, radius):
    tmp = tf.map_fn(
        lambda r: gaussian_laplace(imgs, r),
        radius,
        fn_output_signature=tf.float32,
    )
    return tf.transpose(tmp, (1, 0, 2, 3))
