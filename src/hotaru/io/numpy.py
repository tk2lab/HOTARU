import numpy as np
import tensorflow as tf


def save_numpy(path, val):
    with tf.io.gfile.GFile(path, "bw") as fp:
        np.save(fp, val)


def load_numpy(path):
    with tf.io.gfile.GFile(path, "br") as fp:
        return np.load(fp)
