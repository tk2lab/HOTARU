import numpy as np
import pandas as pd
import tensorflow as tf

from ..filter.laplace import gaussian_laplace_multi
from ..util.dataset import unmasked
from ..util.distribute import ReduceOp
from ..util.distribute import distributed
from ..util.progress import Progress


@distributed(ReduceOp.STACK, ReduceOp.STACK, ReduceOp.STACK)
@tf.function(input_signature=[
    (
        tf.TensorSpec([None], tf.int64),
        tf.TensorSpec([None, None, None], tf.float32),
    ),
    tf.TensorSpec([None, None], tf.bool),
    tf.TensorSpec([None], tf.float32),
])
def _find(args, mask, radius):
    nr = tf.size(radius, out_type=tf.int64)
    h, w = tf.unstack(tf.shape(mask, out_type=tf.int64))
    times, imgs = args
    gl = gaussian_laplace_multi(imgs, radius)
    max_gl = tf.nn.max_pool3d(
        gl[..., None], [1, 3, 3, 3, 1], [1, 1, 1, 1, 1], padding="SAME"
    )[..., 0]
    bit = tf.equal(gl, max_gl) & mask
    glp = tf.where(bit, gl, tf.cast(0, gl.dtype))
    glp = tf.reshape(glp, (-1, h, w))
    pos = tf.math.argmax(glp)
    t = tf.gather(times, pos // nr)
    r = pos % nr
    g = tf.reduce_max(glp, axis=0)
    return t, r, g


def find_peak(imgs, mask, radius, shard=1, batch=1):
    """"""

    nt, h, w = imgs.shape

    mask_t = tf.convert_to_tensor(mask, tf.bool)
    radius_t = tf.convert_to_tensor(radius, tf.float32)

    imgs = Progress(imgs.enumerate(), "find peak", nt, unit="frame", shard=shard, batch=batch)
    t, r, g = _find(imgs, mask, radius)

    idx = tf.math.argmax(g, axis=0, output_type=tf.int32)
    x, y = tf.meshgrid(tf.range(w), tf.range(h))
    idx = tf.reshape(tf.stack((idx, y, x), axis=2), (-1, 3))

    t = tf.gather_nd(t, idx).numpy()
    r = tf.gather_nd(r, idx).numpy()
    g = tf.gather_nd(g, idx).numpy()
    x = tf.reshape(x, -1).numpy()
    y = tf.reshape(y, -1).numpy()

    info = pd.DataFrame(dict(t=t, x=x, y=y, radius=radius[r], intensity=g))
    info.sort_values("intensity", ascending=False, inplace=True)
    info.reset_index(drop=True, inplace=True)
    return info
