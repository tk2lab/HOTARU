import tensorflow.keras.backend as K
import tensorflow as tf


def get_normalized_val(img, pos):
    val = tf.gather_nd(img, pos)
    min_val = K.min(val)
    max_val = K.max(val)
    return (val - min_val) / (max_val - min_val)


def get_magnitude(img, pos):
    val = tf.gather_nd(img, pos)
    return K.max(val) - K.min(val)


class ToDense(object):

    def __init__(self, mask):
        mask = K.constant(mask, tf.bool)
        nx = tf.cast(tf.math.count_nonzero(mask), tf.int32)
        ids = tf.cast(tf.where(mask), tf.int32)
        rmap = tf.scatter_nd(ids, tf.range(nx) + 1, tf.shape(mask)) - 1
        self.rmap = rmap
        self.nx = nx

    def __call__(self, pos, val):
        pos = tf.gather_nd(self.rmap, pos)
        out = tf.scatter_nd(pos[:, None], val, (self.nx,))
        return out
