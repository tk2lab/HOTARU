import tensorflow as tf


def save_tfrecord(path, data):
    with tf.io.TFRecordWriter(path) as writer:
        for d in data:
            writer.write(tf.io.serialize_tensor(d).numpy())


def load_tfrecord(path):
    data = tf.data.TFRecordDataset(path)
    data = data.map(lambda x: tf.io.parse_tensor(x, tf.float32))
    return data
