import tensorflow as tf
import pandas as pd


def save_csv(base, peak, name):
    peak_dict = {k: v for k, v in zip(name, peak)}
    with tf.io.gfile.GFile(base + '.csv', 'w') as fp:
        pd.DataFrame(peak_dict).to_csv(fp)
