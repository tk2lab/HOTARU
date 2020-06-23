import tensorflow as tf
import pickle


def load_pickle(base):
    with tf.io.gfile.GFile(f'{base}.pickle', 'rb') as fp:
        return pickle.load(fp)


def save_pickle(base, val):
    with tf.io.gfile.GFile(f'{base}.pickle', 'wb') as fp:
        pickle.dump(val, fp)
