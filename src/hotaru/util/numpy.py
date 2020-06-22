import tensorflow as tf
import numpy as np


def save_numpy(base, val):
    with tf.io.gfile.GFile(f'{base}.npy', 'bw') as fp:
        np.save(fp, val)


def load_numpy(base):
    with tf.io.gfile.GFile(f'{base}.npy', 'br') as fp:
        return np.load(fp)
