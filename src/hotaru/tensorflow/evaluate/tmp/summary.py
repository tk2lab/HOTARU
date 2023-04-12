import numpy as np
import tensorflow as tf
from matplotlib import cm


def normalized_and_sort(val):
    mag = val.max(axis=1)
    idx = np.argsort(mag)[::-1]
    val = val / mag[:, None]
    val = val[idx]
    return val, mag


def write_spike_summary(writer, val, step=0):
    with writer.as_default():
        val, mag = normalized_and_sort(val)
        spc = val.mean(axis=1)
        spike_train = cm.get_cmap("Reds")(val)[None, ...]
        tf.summary.histogram("max", mag, step=step)
        tf.summary.histogram("avg", spc, step=step)
        tf.summary.image("spike", spike_train, step=step)


def write_footprint_summary(writer, val, mask, step=0):
    with writer.as_default():
        val, mag = normalized_and_sort(val)
        spc = val.mean(axis=1)
        imgs_max = np.zeros(mask.shape)
        imgs_max[mask] = val.max(axis=0) / val.max()
        imgs_max = cm.get_cmap("Greens")(imgs_max)
        tf.summary.histogram("max", mag, step=step)
        tf.summary.histogram("avg", spc, step=step)
        tf.summary.image(f"footprint", imgs_max[None, ...], step=step)
