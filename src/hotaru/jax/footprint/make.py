from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

from ..filter.laplace import gaussian_laplace_multi
from ..utils.saver import SaverMixin
from .normalized import apply_to_normalized
from .segment import get_segment_mask

global_buffer = 2**30


def make_segment_batch(
    imgs, ts, rs, ys, xs, stats=None, buffer=None, num_devices=None, pbar=None
):
    if num_devices is None:
        num_devices = jax.local_device_count()
    if buffer is None:
        buffer = global_buffer
    nt, h, w = imgs.shape
    radius, rs = np.unique(np.array(rs), return_inverse=True)
    idx = np.argsort(radius)
    rmap = np.zeros_like(rs)
    for i, j in enumerate(idx):
        rmap[j] = i
    radius = radius[idx]
    rs = rmap[rs]
    nr = len(radius)
    size = 4 * 4 * nr * h * w
    batch = (buffer + size - 1) // size

    def apply(imgs, mask):
        return gaussian_laplace_multi(imgs, radius, -3)

    def finish(glp):
        return jnp.concatenate(glp, axis=0)

    @jax.pmap
    @jax.vmap
    def make(g, y, x):
        seg = get_segment_mask(g, y, x, mask)
        dmin = jnp.where(seg, g, jnp.inf).min()
        dmax = jnp.where(seg, g, -jnp.inf).max()
        return jnp.where(seg, (g - dmin) / (dmax - dmin), 0)

    nt, y0, x0, mask, avgx, avgt, std0, min0, max0 = stats
    gen = apply_to_normalized(apply, finish, imgs, stats, batch, num_devices, pbar)
    ids = []
    out = []
    for t0, gl in gen:
        t1 = t0 + gl.shape[0]
        idx = jnp.where((t0 <= ts) & (ts < t1))[0]
        t = ts[idx] - t0
        r = rs[idx]
        y = ys[idx]
        x = xs[idx]
        g = gl[t, r]
        ids.append(idx)
        num = y.size
        mod = num % num_devices
        if mod != 0:
            num += 1
            y = jnp.pad(y, [[0, num_devices - mod]])
            x = jnp.pad(x, [[0, num_devices - mod]])
            g = jnp.pad(g, [[0, num_devices - mod], [0, 0], [0, 0]])
        y = y.reshape(num_devices, -1)
        x = x.reshape(num_devices, -1)
        g = g.reshape(num_devices, -1, h, w)
        seg = jnp.concatenate(make(g, y, x), axis=0)
        if mod != 0:
            seg = seg[:-mod]
        out.append(seg)
    return jnp.concatenate(out, axis=0)[jnp.concatenate(ids, axis=0)]
