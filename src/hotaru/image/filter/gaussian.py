import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def gaussian(imgs, r):
    sqrt_2pi = K.constant(np.sqrt(2.0 * np.pi))
    mr = tf.math.ceil(r)
    d = K.square(K.arange(-4.0 * mr, 4.0 * mr + 1.0, 1.0))
    r2 = K.square(r)
    o0 = K.exp(-0.5 * d / r2) / r / sqrt_2pi
    tmp = imgs[..., None]
    tmp = K.conv2d(tmp, tf.reshape(o0, (1, -1, 1, 1)), (1, 1), 'same')
    tmp = K.conv2d(tmp, tf.reshape(o0, (-1, 1, 1, 1)), (1, 1), 'same')
    return tmp[..., 0]
