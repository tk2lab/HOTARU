import tensorflow as tf


def normalized(data, avgt, avgx, std):

    def normalize(img, avgt):
        return (img - avgt - avgx) / std

    avgt = tf.data.Dataset.from_tensor_slices(avgt)
    zip_data = tf.data.Dataset.zip((data, avgt))
    return zip_data.map(normalize)


def masked(data, mask):
    return data.map(lambda x: tf.boolean_mask(x, mask))


def unmasked(data, mask):
    rmap = tf.cast(tf.where(mask), tf.int32)
    shape = tf.shape(mask)
    return data.map(lambda x: tf.scatter_nd(rmap, x, shape))
