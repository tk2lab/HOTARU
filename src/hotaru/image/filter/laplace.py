import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def gaussian_laplace(imgs, r):
    sqrt_2pi = K.constant(np.sqrt(2.0 * np.pi))
    mr = tf.math.ceil(r)
    d = K.square(K.arange(-4.0 * mr, 4.0 * mr + 1.0, 1.0))
    r2 = K.square(r)
    o0 = K.exp(-0.5 * d / r2) / r / sqrt_2pi
    o2 = (1.0 - d / r2) * o0
    tmp = imgs[..., None]
    gl1 = K.conv2d(tmp, tf.reshape(o2, (1, -1, 1, 1)), (1, 1), 'same')
    gl1 = K.conv2d(gl1, tf.reshape(o0, (-1, 1, 1, 1)), (1, 1), 'same')
    gl2 = K.conv2d(tmp, tf.reshape(o2, (-1, 1, 1, 1)), (1, 1), 'same')
    gl2 = K.conv2d(gl2, tf.reshape(o0, (1, -1, 1, 1)), (1, 1), 'same')
    return (gl1 + gl2)[..., 0]


def gaussian_laplace_multi(imgs, radius):
    tmp = K.map_fn(lambda r: gaussian_laplace(imgs, r), radius)
    return tf.transpose(tmp, (1, 0, 2, 3))
