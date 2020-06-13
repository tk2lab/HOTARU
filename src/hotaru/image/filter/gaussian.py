import tensorflow as tf
from numpy import pi as PI


def gaussian(imgs, r):
    mr = tf.math.ceil(r)
    d = tf.square(tf.range(-4.0 * mr, 4.0 * mr + 1.0, 1.0))
    r2 = tf.square(r)
    o0 = tf.exp(-0.5 * d / r2) / tf.sqrt(2.0 * PI) / r
    tmp = imgs[..., None]
    tmp = tf.nn.conv2d(tmp, tf.reshape(o0, (1, -1, 1, 1)), (1, 1), 'SAME')
    tmp = tf.nn.conv2d(tmp, tf.reshape(o0, (-1, 1, 1, 1)), (1, 1), 'SAME')
    return tmp[..., 0]
