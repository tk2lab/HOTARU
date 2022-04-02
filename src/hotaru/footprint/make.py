import tensorflow as tf
import numpy as np
import click

from ..util.distribute import distributed
from ..util.distribute import ReduceOp
from ..util.dataset import unmasked
from ..image.filter.laplace import gaussian_laplace_multi

from .segment import get_segment


def make_segment(dataset, mask, peaks, batch, verbose):

    @distributed(ReduceOp.CONCAT)
    def _make(data, mask, index, ts, rs, ys, xs, radius):
        idx, imgs = data
        idx = tf.cast(idx, tf.int32)
        cond = (idx[0] <= ts) & (ts <= idx[-1])
        ids = tf.cast(tf.where(cond)[:, 0], tf.int32)
        il = tf.gather(index, ids)
        tl = tf.gather(ts, ids) - idx[0]
        rl = tf.gather(rs, ids)
        yl = tf.gather(ys, ids)
        xl = tf.gather(xs, ids)
        gls = gaussian_laplace_multi(imgs, radius)
        gl = tf.gather_nd(gls, tf.stack([tl, rl], 1))

        out = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for k in tf.range(tf.size(ids)):
            idx, g, y, x = il[k], gl[k], yl[k], xl[k]
            seg = get_segment(g, y, x, mask)
            sg = tf.boolean_mask(g, seg)
            gmin = tf.math.reduce_min(sg)
            gmax = tf.math.reduce_max(sg)
            out = out.write(k, tf.boolean_mask((g - gmin) / (gmax - gmin) * tf.cast(seg, tf.float32), mask))
        return out.stack(),

    nk = peaks.shape[0]
    index = tf.convert_to_tensor(peaks.index.values, tf.int32)
    ts = tf.convert_to_tensor(peaks['t'].values, tf.int32)
    xs = tf.convert_to_tensor(peaks['x'].values, tf.int32)
    ys = tf.convert_to_tensor(peaks['y'].values, tf.int32)
    rs = tf.convert_to_tensor(peaks['radius'].values, tf.float32)
    radius, rs = tf.unique(rs)

    dataset = unmasked(dataset, mask)
    dataset = dataset.enumerate().batch(batch)

    with click.progressbar(length=nk, label='Make') as prog:
        segment, = _make(dataset, mask, index, ts, rs, ys, xs, radius, prog=prog)
    return segment.numpy()
