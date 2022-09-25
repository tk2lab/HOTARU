import numpy as np
import tensorflow as tf


def remove_noise(val, scale=100):
    num_bins = (tf.size(val) // scale) + 1
    hist = tf.histogram_fixed_width(val, [0.0, 1.0], num_bins)
    hist_pos = tf.math.argmax(hist)
    thr = tf.cast(hist_pos, tf.float32) / tf.cast(num_bins, tf.float32)
    return tf.where(val >= thr, (val - thr) / (1 - thr), 0)


def get_segment(gl, y, x, mask):
    pos = get_segment_index(gl, y, x, mask)
    npos = tf.shape(pos)[0]
    return tf.scatter_nd(pos, tf.ones((npos,), tf.bool), tf.shape(mask))


def get_segment_index(gl, y, x, mask):
    return tf.numpy_function(
        get_segment_index_py,
        [gl, y, x, mask],
        tf.int32,
    )


def get_segment_py(img, y0, x0, mask):
    pos = get_segment_index_py(img, y0, x0, mask)
    h, w = img.shape
    out = np.zeros((h, w), bool)
    out[pos[:, 0], pos[:, 1]] = True
    return out


def get_segment_index_py(img, y0, x0, mask):
    delta = [
        (dy, dx)
        for dy in [-1, 0, 1]
        for dx in [-1, 0, 1]
        if (dy, dx) != (0, 0)
    ]
    mask = np.pad(mask, [[0, 1], [0, 1]])
    pos = np.zeros_like(mask, bool)
    pos = np.pad(pos, [[0, 1], [0, 1]], constant_values=True)

    pos[y0 + 1, x0 + 1] = True
    q = []
    for dy, dx in delta:
        q.append((y0 + dy, x0 + dx, -dy, -dx))

    while len(q) > 0:
        yt, xt, dy, dx = q.pop()
        yo, xo = yt + dy, xt + dx
        if mask[yt + 1, xt + 1] and (0 < img[yt, xt] <= img[yo, xo]):
            pos[yt + 1, xt + 1] = True
            q = [(y, x, dy, dx) for y, x, dy, dx in q if (y, x) != (yt, xt)]
            for dy, dx in delta:
                if not pos[yt + dy + 1, xt + dx + 1]:
                    q.append([yt + dy, xt + dx, -dy, -dx])

    return np.stack(np.where(pos), axis=1).astype(np.int32)


def get_segment_index_tf(img, y0, x0, mask):
    def push(y, x, dy, dx):
        yn, xn = y + dy, x + dx
        if mask[yn, xn]:
            if img[yn, xn] <= img[y, x]:
                q.enqueue([yn, xn, dy, dx])

    mask = tf.pad(mask, [[0, 1], [0, 1]])
    pos = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

    q = tf.queue.FIFOQueue(2000, [tf.int32] * 4, [()] * 4)
    i = tf.constant(0)
    pos = pos.write(i, [y0, x0])
    for dy in tf.constant([-1, 0, 1]):
        for dx in tf.constant([-1, 0, 1]):
            if tf.not_equal(dx, 0) & tf.not_equal(dy, 0):
                push(y0, x0, dy, dx)

    while q.size() > 0:
        y, x, dy, dx = q.dequeue()
        i += 1
        pos = pos.write(i, [y, x])
        push(y, x, dy, dx)
        if dx == 0:
            push(y, x, dy, +1)
            push(y, x, dy, -1)
        elif dy == 0:
            push(y, x, +1, dx)
            push(y, x, -1, dx)

    return pos.stack()
