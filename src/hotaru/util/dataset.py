import tensorflow.keras.backend as K
import tensorflow as tf


def normalized(data, avgt, avgx, std):

    def normalize(img, avgt):
        return (img - avgt - avgx) / std

    avgt = tf.data.Dataset.from_tensor_slices(avgt)
    avgx = K.constant(avgx)
    std = K.constant(std)
    zip_data = tf.data.Dataset.zip((data, avgt))
    return zip_data.map(normalize)


def masked(data, mask):
    mask = K.constant(mask, tf.bool)
    return data.map(lambda x: tf.boolean_mask(x, mask))


def unmasked(data, mask):
    mask = K.constant(mask, tf.bool)
    rmap = tf.cast(tf.where(mask), tf.int32)
    shape = tf.shape(mask)
    return data.map(lambda x: tf.scatter_nd(rmap, x, shape))
