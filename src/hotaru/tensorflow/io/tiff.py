import tensorflow as tf
import tifffile


def load_tiff(path):
    with tf.io.gfile.GFile(path, "rb") as fp:
        return tifffile.imload(fp)


def save_tiff(path, val):
    with tf.io.gfile.GFile(path, "wb") as fp:
        tifffile.imwrite(fp, val)
