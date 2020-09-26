import tensorflow as tf
import numpy as np

from ..util.gs import ensure_local_file


def save_numpy(base, val):
    with tf.io.gfile.GFile(f'{base}.npy', 'bw') as fp:
        np.save(fp, val)


def load_numpy(base, mmap=False):
    filename = f'{base}.npy'
    if mmap:
        filename = ensure_local_file(filename)
        return np.load(filename, mmap_mode='r')
    else:
        with tf.io.gfile.GFile(filename, 'br') as fp:
            return np.load(fp)
