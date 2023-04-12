import jax
import jax.numpy as jnp
import numpy as np

from ...io.saver import (
    load,
    save,
)
from ..filter.laplace import gaussian_laplace_multi
from ..filter.map import mapped_imgs
from .segment import get_segment_mask


def make_segment_batch(imgs, stats, ts, rs, ys, xs, pbar=None):
    print(rs)
    radius, rs = np.unique(np.array(rs), return_inverse=True)
    print(radius)
    print(rs)
    idx = np.argsort(radius)
    radius = radius[idx]
    rmap = np.zeros_like(rs)
    for i, j in enumerate(idx):
        rmap[j] = i
    rs = rmap[rs]
    nr = len(radius)

    nt, x0, y0, mask, avgx, avgt, std0 = stats
    h, w = mask.shape
    imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
    print(nr, y0, x0, h, w)

    scale = 20 * nr

    def apply(t0, imgs):
        avgt = imgs.mean(axis=0, keepdims=True)
        imgs = (imgs - avgx - avgt) / std0
        return t0, gaussian_laplace_multi(imgs, radius, -3)

    def aggregate(t0, gl):
        num_devices = t0.size
        gl = jnp.concatenate(gl, axis=0)
        t0 = t0[0]
        t1 = t0 + gl.shape[0]
        cond = (t0 <= ts) & (ts < t1)
        idx = jnp.where(cond)[0]
        t = ts[idx] - t0
        r = rs[idx]
        y = ys[idx]
        x = xs[idx]
        g = gl[t, r]
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
        return idx, seg

    def finish(idx, seg):
        seg = jnp.concatenate(seg, axis=0)
        idx = jnp.concatenate(idx, axis=0)
        rmap = jnp.empty(idx.size, jnp.int32)
        for i, j in enumerate(idx):
            rmap[j] = i
        return seg[rmap]

    @jax.pmap
    @jax.vmap
    def make(g, y, x):
        seg = get_segment_mask(g, y, x, mask)
        dmin = jnp.where(seg, g, jnp.inf).min()
        dmax = jnp.where(seg, g, -jnp.inf).max()
        return jnp.where(seg, (g - dmin) / (dmax - dmin), 0)

    return mapped_imgs(imgs, apply, aggregate, finish, scale, pbar)
