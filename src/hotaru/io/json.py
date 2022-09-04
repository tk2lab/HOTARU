import json

import tensorflow as tf


def load_json(path):
    with tf.io.gfile.GFile(path, "rb") as fp:
        return json.load(fp)


def save_json(path, val):
    with tf.io.gfile.GFile(path, "wb") as fp:
        json.dump(val, fp)
