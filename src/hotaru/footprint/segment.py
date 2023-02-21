import numpy as np
import tensorflow as tf


def get_segment_mask(gl, y, x, mask):
    return tf.numpy_function(
        get_segment_mask_py,
        [gl, y, x, mask],
        tf.bool,
    )


def get_segment_mask_py(img, y0, x0, mask):
    delta = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if (dy, dx) != (0, 0)]
    mask = np.pad(mask, [[0, 1], [0, 1]])
    pos = np.zeros_like(mask, bool)

    pos[y0, x0] = True
    q = []
    for dy, dx in delta:
        y1, x1 = y0 + dy, x0 + dx
        if mask[y1, x1]:
            q.append((y1, x1, dy, dx))

    while len(q) > 0:
        yt, xt, dy, dx = q.pop()
        yo, xo = yt - dy, xt - dx
        if 0 < img[yt, xt] <= img[yo, xo]:
            pos[yt, xt] = True
            q = [(y, x, dy, dx) for y, x, dy, dx in q if (y, x) != (yt, xt)]
            for dy, dx in delta:
                yn, xn = yt + dy, xt + dx
                if not pos[yn, xn] and mask[yn, xn]:
                    q.append([yn, xn, dy, dx])

    return pos[:-1, :-1]
