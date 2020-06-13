import tensorflow as tf


def make_tfrecord(filename, data, prog=None):
    with tf.io.TFRecordWriter(filename) as writer:
        for d in data:
            writer.write(tf.io.serialize_tensor(d).numpy())
            if prog is not None:
                prog.add(1)
