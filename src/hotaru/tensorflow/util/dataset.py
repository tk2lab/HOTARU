import numpy as np
import tensorflow as tf


def clipped(data, y0, y1, x0, x1):
    data = data.map(lambda img: img[y0:y1, x0:x1])
    data.shape = data.shape[0], y1 - y0, x1 - x0


def normalized(data, avg0, avgx, avgt, std):
    shape = data.shape
    avgx = tf.convert_to_tensor(avgx, tf.float32)
    avgt = tf.convert_to_tensor(avgt, tf.float32)
    std = tf.convert_to_tensor(std, tf.float32)
    data = data.enumerate().map(lambda t, dat: (dat - avg0 - avgx - avgt[t]) / std)
    data.shape = shape
    return data


def normalized_masked_image(imgs, mask, avg0, avgx, avgt, std):
    nt, h, w = imgs.shape
    mask = tf.convert_to_tensor(mask / std, tf.float32)
    avgx = tf.convert_to_tensor(avgx, tf.float32)
    avgt = tf.convert_to_tensor(avgt, tf.float32)
    index = tf.where(mask)
    avgx = tf.scatter_nd(index, avgx, [h, w])
    imgs = imgs.enumerate().map(lambda t, img: mask * (img - avg0 - avgx - avgt[t]))
    imgs.shape = nt, h, w
    return imgs


def masked(imgs, mask):
    nt = imgs.shape[0]
    nx = np.count_nonzero(mask)
    mask = tf.convert_to_tensor(mask, tf.bool)
    data = imgs.map(lambda x: tf.boolean_mask(x, mask))
    data.shape = nt, nx
    return data


def unmasked(data, mask):
    nt = data.shape[0]
    h, w = mask.shape
    mask = tf.convert_to_tensor(mask, tf.bool)
    rmap = tf.cast(tf.where(mask), tf.int32)
    imgs = data.map(lambda x: tf.scatter_nd(rmap, x, [h, w]))
    imgs.shape = nt, h, w
    return imgs
