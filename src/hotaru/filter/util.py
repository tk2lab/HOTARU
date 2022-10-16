import tensorflow as tf


def erosion(x):
    y = x[1:-1, 1:-1]
    y &= x[0:-2, 1:-1]
    y &= x[2:, 1:-1]
    y &= x[1:-1, 0:-2]
    y &= x[1:-1, 2:]
    y &= x[0:-2, 0:-2]
    y &= x[2:, 0:-2]
    y &= x[0:-2, 0:-2]
    y &= x[2:, 2:]
    return tf.pad(y, [[1, 1], [1, 1]])
