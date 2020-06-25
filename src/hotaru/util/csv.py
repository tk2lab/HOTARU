import tensorflow as tf
import numpy as np


def load_csv(base):
    with tf.io.gfile.GFile(f'{base}.csv', 'r') as fp:
        return np.loadtxt(fp, delimiter=',', skiprows=1)


def save_csv(base, data, fmt, header):
    with tf.io.gfile.GFile(f'{base}.csv', 'w') as fp:
        np.savetxt(fp, data, fmt, ',', header=header)
