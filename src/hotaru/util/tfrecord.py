import tensorflow as tf
import numpy as np

from .numpy import load_numpy, save_numpy
from .pickle import load_pickle, save_pickle


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


def moving_average_imgs(data, window, shift):

    def _gen():
        masked_imgs = load_numpy(f'{data}-data', mmap=True)
        h, w = mask.shape
        for t in range(window, nt, shift):
            img = np.zeros((h, w), np.float32)
            img[mask] = masked_imgs[t-window:t].mean(axis=0)
            yield img

    mask = load_numpy(f'{data}-mask')
    nt = load_pickle(f'{data}-stat')[1]
    nnt = (nt - window + 1) // shift
    return tf.data.Dataset.from_generator(_gen, tf.float32), mask, nnt
