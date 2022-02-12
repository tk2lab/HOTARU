import tensorflow as tf
from tqdm import trange


def save_tfrecord(filebase, data, nt=None, verbose=1):
    with tf.io.TFRecordWriter(f'{filebase}.tfrecord') as writer:
        with trange(nt, desc='Save', disable=verbose == 0) as prog:
            for d in data:
                writer.write(tf.io.serialize_tensor(d).numpy())
                if prog is not None:
                    prog.update(1)


def load_tfrecord(filebase):
    data = tf.data.TFRecordDataset(f'{filebase}.tfrecord')
    data = data.map(lambda ex: tf.io.parse_tensor(ex, tf.float32))
    return data
