import pickle

import tensorflow as tf


def load_pickle(path):
    with tf.io.gfile.GFile(path, 'rb') as fp:
        return pickle.load(fp)


def save_pickle(path, val):
    with tf.io.gfile.GFile(path, 'wb') as fp:
        pickle.dump(val, fp)
