import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from ..filter.laplace import gaussian_laplace_multi
from ..filter.util import erosion
from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from ..util.dataset import unmasked
from ..util.progress import Progress
from .segment import get_segment_mask


@distributed(ReduceOp.CONCAT, ReduceOp.CONCAT)
@tf.function(input_signature=[tf.TensorSpec((None, None), tf.float32)])
def _scale(data):
    top, index = tf.math.top_k(data, 2)
    scale = top[:, 1]
    index = index[:, 0]
    index = tf.stack([tf.range(tf.size(index)), index], axis=1)
    data = tf.tensor_scatter_nd_update(data, index, scale)
    data /= tf.where(scale > 0, scale, 1.0)[:, None]
    return data, scale


@distributed(
    ReduceOp.CONCAT,
    ReduceOp.CONCAT,
    ReduceOp.CONCAT,
    ReduceOp.CONCAT,
    ReduceOp.CONCAT,
)
@tf.function(input_signature=[
    tf.TensorSpec([None, None, None], tf.float32),
    tf.TensorSpec([None], tf.float32),
])
def _filter(imgs, radius_list):
    nr = tf.size(radius_list)
    nk, h, w = tf.unstack(tf.shape(imgs))
    logs = gaussian_laplace_multi(imgs, radius_list)
    flat = tf.reshape(logs, (nk, nr * h * w))
    peak = tf.math.argmax(flat, axis=1, output_type=tf.int32)
    r = peak // (h * w)
    y = peak % (h * w) // w
    x = peak % w
    logs = tf.gather(logs, r, batch_dims=1)
    radius = tf.gather(radius_list, r)
    firmness = tf.math.reduce_max(flat, axis=1)
    return logs, y, x, radius, firmness


@distributed(ReduceOp.CONCAT, ReduceOp.CONCAT)
@tf.function(input_signature=[
    (
        tf.TensorSpec([None], tf.float32),
        tf.TensorSpec([None, None], tf.float32),
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.int32),
    ),
    tf.TensorSpec([None, None], tf.bool),
])
def _segment(args, mask):
    dat, log, y, x = args
    seg = get_segment_mask(log, y, x, mask)
    seg = tf.ensure_shape(seg, mask.shape)
    contour = tf.boolean_mask(~erosion(seg) & seg, mask)
    seg = tf.boolean_mask(seg, mask)
    dmin = tf.reduce_max(tf.boolean_mask(dat, contour))
    dmax = tf.reduce_max(dat)
    scale = dmax - dmin
    dat = tf.where(seg & (dat > dmin), (dat - dmin) / scale, 0)
    return dat[None, ...], scale[None, ...]


def clean_segment(data, mask, radius, batch):
    """"""
    h, w = mask.shape
    nr = radius.size
    nk = data.shape[0]

    mask = tf.convert_to_tensor(mask, tf.bool)
    radius_list = tf.convert_to_tensor(radius, tf.float32)

    data_ds = tf.data.Dataset.from_tensor_slices(data)
    data_ds = Progress(data_ds, "scale", nk, unit="cell", batch=batch)
    data, scale1 = _scale(data_ds)

    data_ds = tf.data.Dataset.from_tensor_slices(data)
    data_ds.shape = nk, h, w
    imgs_ds = unmasked(data_ds, mask)
    imgs_ds = Progress(imgs_ds, "filter", nk, unit="cell", batch=batch)
    logs, y, x, radius, firmness = _filter(imgs_ds, radius_list)

    logs_ds = tf.data.Dataset.from_tensor_slices((data, logs, y, x))
    logs_ds = Progress(logs_ds, "segment", nk, unit="cell")
    data, scale2 = _segment(logs_ds, mask)

    scale = scale1 * scale2
    return data, scale, x.numpy(), y.numpy(), radius.numpy(), firmness.numpy()
