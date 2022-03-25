import tensorflow as tf
import pandas as pd


def load_csv(path):
    with tf.io.gfile.GFile(path, 'r') as fp:
        return pd.read_csv(fp, index_col=0)


def save_csv(path, data):
    with tf.io.gfile.GFile(path, 'w') as fp:
        data.to_csv(fp)
