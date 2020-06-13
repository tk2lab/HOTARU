import tensorflow as tf
import pandas as pd
import numpy as np


def save_csv(base, peak, name):
    peak_dict = {k: v for k, v in zip(name, peak)}
    with tf.io.gfile.GFile(f'{base}.csv', 'w') as fp:
        pd.DataFrame(peak_dict).to_csv(fp)


def load_csv(base, name, typ):
    with tf.io.gfile.GFile(f'{base}.csv', 'r') as fp:
        peak = pd.read_csv(fp)
        peak = tuple(np.array(peak[k], t) for k, t in zip(name, typ))
    return peak
