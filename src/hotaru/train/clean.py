import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from ..filter.laplace import gaussian_laplace_multi
from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from ..util.progress import Progress


def unmask(data, mask):
    data = tf.convert_to_tensor(data, tf.float32)
    mask = tf.convert_to_tensor(mask, tf.float32)
    nk = tf.shape(data)[0]
    h, w = tf.shape(mask)
    index = tf.where(mask)
    data = tf.transpose(data)
    imgs = tf.scatter_nd(index, data, (h, w, nk))
    imgs = tf.transpose(imgs, (2, 0, 1))
    return imgs


def clean(data, mask, bins, radius, batch):
    """"""
    radius_list = tf.convert_to_tensor(radius, tf.float32)
    imgs = unmask(data, mask)
    nk, h, w = imgs.shape

    def _scale(imgs):
        nk, h, w = tf.unstack(tf.shape(imgs))
        flat = tf.reshape(imgs, (nk, h * w))
        top, index = tf.math.top_k(flat, 2)
        scale = top[:, 1]
        index = index[:, 0]
        k = tf.range(nk)
        y = index // w
        x = index % w
        index = tf.stack([k, y, x], axis=1)
        imgs = tf.tensor_scatter_nd_update(imgs, index, scale)
        imgs /= tf.where(scale > 0, scale, 1.0)[:, None, None]
        return imgs, scale

    imgs, scale = _scale(imgs)

    @distributed(
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        ReduceOp.CONCAT,
        input_signature=[tf.TensorSpec((None, h, w), tf.float32)],
    )
    def _filter(imgs):
        logs = gaussian_laplace_multi(imgs, radius_list)
        nk, nr, h, w = tf.unstack(tf.shape(logs))
        flat = tf.reshape(logs, (nk, nr * h * w))
        peak = tf.math.argmax(flat, axis=1, output_type=tf.int32)
        r = peak // (h * w)
        y = peak % (h * w) // w
        x = peak % w
        logs = tf.gather(logs, r, batch_dims=1)
        radius = tf.gather(radius_list, r)
        firmness = tf.math.reduce_max(flat, axis=1)

        out = tf.TensorArray(tf.float32, size=tf.size(ids), element_shape=[nx])
        for k in tf.range(tf.size(ids)):
            img, y, x = gl[k], yl[k], xl[k]
            seg = get_segment_mask(img, y, x, mask)
            pos = tf.where(seg)
            val = tf.gather_nd(img, pos)
            val /= tf.math.reduce_max(val)
            img = tf.scatter_nd(pos, val, [h, w])
            dat = tf.boolean_mask(img, mask)
            out = out.write(k, dat)
        return ids, out.stack()
        return logs, y, x, radius, firmness

    imgs_ds = tf.data.Dataset.from_tensor_slices(imgs)
    imgs_ds = Progress(imgs_ds, "filter", nk, batch=batch)
    logs, y, x, radius, firmness = _filter(imgs_ds)

    @distributed(ReduceOp.CONCAT)
    def _label(log):
        return tfa.image.connected_components(log > 0)[None, ...]

    logs = tf.data.Dataset.from_tensor_slices(logs)
    logs = Progress(logs, "label", nk)
    label = _label(logs)

    target_id = tf.gather_nd(label, tf.stack([y, x], axis=1), batch_dims=1)
    roi = label == target_id[:, None, None]
    imgs = tf.where(roi, imgs, 0)
    imin = tf.math.reduce_min(imgs, axis=(1, 2))
    imgs = (imgs - imin[:, None, None]) / (1 - imin[:, None, None])
    scale *= (1 - imin)
    data = tf.boolean_mask(imgs, mask, axis=1)
    return data.numpy(), scale.numpy(), x.numpy(), y.numpy(), radius.numpy(), firmness.numpy()
