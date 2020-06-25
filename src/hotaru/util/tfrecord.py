import tensorflow as tf


def save_tfrecord(filebase, data, nt=None, verbose=1):
    prog = tf.keras.utils.Progbar(nt, verbose=verbose)
    with tf.io.TFRecordWriter(f'{filebase}.tfrecord') as writer:
        for d in data:
            writer.write(tf.io.serialize_tensor(d).numpy())
            if prog is not None:
                prog.add(1)


def load_tfrecord(filebase):
    data = tf.data.TFRecordDataset(f'{filebase}.tfrecord')
    data = data.map(lambda ex: tf.io.parse_tensor(ex, tf.float32))
    return data
