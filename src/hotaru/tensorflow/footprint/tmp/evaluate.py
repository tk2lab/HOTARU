import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from ..filter.laplace import gaussian_laplace_multi
from ..util.distribute import (
    ReduceOp,
    distributed,
)
from ..util.progress import Progress


def unmask(data, mask):
    nk = tf.shape(data)[0]
    h, w = tf.shape(mask)
    index = tf.where(mask)
    data = tf.transpose(data)
    imgs = tf.scatter_nd(index, data, (h, w, nk))
    imgs = tf.transpose(imgs, (2, 0, 1))
    return imgs


def clean(imgs, bins=100):
    def _clean(img, y, x):
        flat = tf.reshape(img, (-1,))
        n = tf.histogram_fixed_width(flat, (0.0, 1.0), bins + 2)
        thr = tf.argmax(n[1:]) / bins
        binary = img > thr
        label = tfa.image.connected_component(binary)
        img = (img - thr) / (1 - thr)
        img = tf.where(label == label[y, x], img, 0)
        return img, thr

    nk, h, w = tf.unstack(tf.shape(imgs))
    flat = tf.reshape(imgs, (nk, h * w))
    top, index = tf.tok_k(flat, 2)
    scale = top[:, 1]
    index = index[:, 0]
    ok = scale > 0
    nk = tf.math.count_nonzer(ok)
    imgs = tf.boolean_mask(imgs, ok)
    scale = tf.boolean_mask(scale, ok)
    index = tf.boolean_mask(index, ok)
    k = tf.range(nk)
    y = index // w
    x = index % w
    index = tf.stack([k, y, x], axis=1)
    imgs = tf.tensor_scatter_nd_update(imgs, index, scale)
    imgs /= scale[:, None, None]
    imgs, thr = tf.vectorized_map(_clean, (imgs, y, x))
    scale *= tf.math.reciprocal_no_nan(1 - thr)
    return ok, imgs, scale


def evaluate_footprint(footprint, mask, radius, batch):
    """"""

    @distributed(ReduceOp.CONCAT, ReduceOp.CONCAT)
    def _evaluate(data, mask, radius):
        imgs = unmask(data, imgss)

        logs = gaussian_laplace_multi(imgs, radius)
        nk, h, w = tf.shape(logs)[0], tf.shape(logs)[2], tf.shape(logs)[3]
        hw = h * w
        lsr = tf.reshape(logs, (nk, -1))
        pos = tf.cast(tf.argmax(lsr, axis=1), tf.int32)
        rs = pos // hw
        ys = (pos % hw) // w
        xs = (pos % hw) % w
        indices = tf.stack([tf.range(nk), rs], axis=1)
        logs = tf.gather_nd(logs, indices)

        nx = tf.size(rs)
        indices = tf.stack([tf.range(nx), ys, xs], axis=1)
        imgmin = tf.math.reduce_min(imgs, axis=(1, 2))
        imgmax = tf.math.reduce_max(imgs, axis=(1, 2))
        logpeak = tf.gather_nd(logs, indices)
        firmness = logpeak / (imgmax - imgmin)
        return firmness, rs, ys, xs

    nk = footprint.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(footprint)
    dataset = dataset.batch(batch)

    mask_t = tf.convert_to_tensor(mask, tf.bool)
    radius_t = tf.convert_to_tensor(radius, tf.float32)

    with Progress(length=nk, label="Eval", unit="footprint") as prog:
        out = _evaluate(dataset, mask_t, radius_t, prog=prog)
    f, r, y, x = (o.numpy() for o in out)
    return dict(x=x, y=y, radius=radius[r], firmness=f)


def clean_footprint(footprint, mask, bins=51):
    """"""

    def threshold(x):
        n, b = np.histogram(x[x > 0], bins=np.linspace(0, 1, bins))
        return b[np.argmax(n)]

    denseness = []
    scale = []
    tmp = np.zeros_like(mask, dtype=xs.dtype)
    for f, y, x in zip(footprint, info.y, inof.x):
        thr = threshold(f)
        fmax = f.max()
        tmp[mask] = f
        obj, n = ndi.label(tmp > thr)
        region = obj == obj[y, x]
        tmp[...] = (tmp - thr) / (fmax - thr)
        tmp[...] = np.where(region, tmp, 0.0)
        f[:] = tmp[mask]
        denseness.append(thr / fmax)
        scale.append(fmax - thr)
    return dict(denseness=denseness, scale=scale)


def calc_area(x, threshold):
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    x = (x - xmin) / (xmax - xmin)
    b = x > threshold
    area = np.count_nonzero(b, axis=1)
    b = b.astype(np.float32)
    c = (b @ b.T) / b.sum(axis=1)
    np.fill_diagonal(c, 0.0)
    overwrap = c.max(axis=1)
    return area, overwrap
