import tensorflow as tf
import numpy as np
import pandas as pd
import click

from ..util.distribute import distributed, ReduceOp
from ..util.dataset import unmasked
from ..image.filter.laplace import gaussian_laplace_multi


def find_peak(data, mask, radius, shard, batch, nt=None, verbose=1):

    @distributed(ReduceOp.STACK, ReduceOp.STACK, ReduceOp.STACK)
    def _find(data, mask, radius):
        times, imgs = data
        times = tf.cast(times, tf.int32)
        gl = gaussian_laplace_multi(imgs, radius)
        max_gl = tf.nn.max_pool3d(
            gl[..., None], [1, 3, 3, 3, 1], [1, 1, 1, 1, 1], padding='SAME'
        )[..., 0]
        bit = tf.equal(gl, max_gl) & mask
        glp = tf.where(bit, gl, tf.cast(0, gl.dtype))
        shape = tf.shape(glp)
        glp = tf.reshape(glp, (-1, shape[2], shape[3]))
        pos = tf.math.argmax(glp, output_type=tf.int32)
        t = tf.gather(times, pos // shape[1])
        r = pos % shape[1]
        g = tf.reduce_max(glp, axis=0)
        return t, r, g

    data = unmasked(data, mask)
    data = data.enumerate().shard(shard, 0).batch(batch)
    mask = tf.convert_to_tensor(mask, tf.bool)
    radius_ = tf.convert_to_tensor(radius, tf.float32)
    total = (nt + shard - 1) // shard
    with click.progressbar(length=total, label='Find') as prog:
        t, r, g = _find(data, mask, radius_, prog=prog)

    idx = tf.math.argmax(g, axis=0, output_type=tf.int32)
    shape = tf.shape(mask)
    x, y = tf.meshgrid(tf.range(shape[1]), tf.range(shape[0]))
    idx = tf.reshape(tf.stack((idx, y, x), axis=2), (-1, 3))
    t = tf.gather_nd(t, idx).numpy()
    r = tf.gather_nd(r, idx).numpy()
    g = tf.gather_nd(g, idx).numpy()
    x = tf.reshape(x, -1).numpy()
    y = tf.reshape(y, -1).numpy()

    idx = np.argsort(g)[::-1]
    idx = idx[g[idx] > 0.0]
    return pd.DataFrame(dict(
        t=t, radius=radius[r], intensity=g, x=x, y=y,
    )).iloc[idx].reset_index(drop=True)
