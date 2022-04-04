import tensorflow as tf


def normalized(data, std, avgt, avgx):

    def normalize(img, avgt):
        return (img - avgt - avgx) / std

    avgt = tf.data.Dataset.from_tensor_slices(avgt)
    avgx = tf.convert_to_tensor(avgx, tf.float32)
    std = tf.convert_to_tensor(std, tf.float32)
    zip_data = tf.data.Dataset.zip((data, avgt))
    return zip_data.map(normalize)


def masked(data, mask):
    mask = tf.convert_to_tensor(mask, tf.bool)
    return data.map(lambda x: tf.boolean_mask(x, mask))


def unmasked(data, mask):
    mask = tf.convert_to_tensor(mask, tf.bool)
    rmap = tf.cast(tf.where(mask), tf.int32)
    shape = tf.shape(mask)
    return data.map(lambda x: tf.scatter_nd(rmap, x, shape))
