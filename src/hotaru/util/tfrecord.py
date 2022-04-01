import tensorflow as tf
import click


def save_tfrecord(path, data, nt=None, verbose=1):
    with tf.io.TFRecordWriter(path) as writer:
        with click.progressbar(length=nt, label='Save') as prog:
            for d in data:
                writer.write(tf.io.serialize_tensor(d).numpy())
                prog.update(1)


def load_tfrecord(path):
    data = tf.data.TFRecordDataset(path)
    data = data.map(lambda x: tf.io.parse_tensor(x, tf.float32))
    return data
