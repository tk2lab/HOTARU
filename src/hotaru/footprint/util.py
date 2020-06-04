import tensorflow as tf


def get_normalized_val(img, pos):
    val = tf.gather_nd(img, pos)
    min_val = tf.reduce_min(val)
    max_val = tf.reduce_max(val)
    return (val - min_val) / (max_val - min_val)


def get_magnitude(img, pos):
    val = tf.gather_nd(img, pos)
    return tf.reduce_max(val) - tf.reduce_min(val)


class ToDense(object):

    def __init__(self, mask):
        nx = tf.cast(tf.math.count_nonzero(mask), tf.int32)
        ids = tf.cast(tf.where(mask), tf.int32)
        rmap = tf.scatter_nd(ids, tf.range(nx) + 1, tf.shape(mask)) - 1
        self.rmap = rmap
        self.nx = nx

    def __call__(self, pos, val):
        pos = tf.gather_nd(self.rmap, pos)
        out = tf.scatter_nd(pos[:, None], val, (self.nx,))
        return out
