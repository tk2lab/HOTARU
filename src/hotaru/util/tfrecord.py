import tensorflow as tf
from tqdm import trange


def save_tfrecord(path, data, nt=None, verbose=1):
    with tf.io.TFRecordWriter(path) as writer:
        with trange(nt, desc='Save', disable=verbose == 0) as prog:
            for d in data:
                writer.write(tf.io.serialize_tensor(d).numpy())
                if prog is not None:
                    prog.update(1)


def load_tfrecord(path):
    data = tf.data.TFRecordDataset(path)
    data = data.map(lambda x: tf.io.parse_tensor(x, tf.float32))
    return data
