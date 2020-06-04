import tensorflow as tf
from numpy import pi as PI


def gaussian_laplace(imgs, r):
    r = tf.convert_to_tensor(r)
    mr = tf.math.ceil(r)
    d = tf.square(tf.range(-4.0 * mr, 4.0 * mr + 1.0, 1.0))
    r2 = tf.square(r)
    o0 = tf.exp(-0.5 * d / r2) / tf.sqrt(2.0 * PI) / r
    o2 = (1.0 - d / r2) * o0
    tmp = imgs[...,tf.newaxis]
    gl1 = tf.nn.conv2d(tmp, tf.reshape(o2, (1, -1, 1, 1)), (1, 1), 'SAME')
    gl1 = tf.nn.conv2d(gl1, tf.reshape(o0, (-1, 1, 1, 1)), (1, 1), 'SAME')
    gl2 = tf.nn.conv2d(tmp, tf.reshape(o2, (-1, 1, 1, 1)), (1, 1), 'SAME')
    gl2 = tf.nn.conv2d(gl2, tf.reshape(o0, (1, -1, 1, 1)), (1, 1), 'SAME')
    return (gl1 + gl2)[..., 0]


def gaussian_laplace_multi(imgs, radius):
    tmp = tf.map_fn(lambda r: gaussian_laplace(imgs, r), radius)
    return tf.transpose(tmp, (1, 2, 3, 0))
