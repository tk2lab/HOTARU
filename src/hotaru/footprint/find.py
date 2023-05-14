from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..filter.laplace import gaussian_laplace_multi
from ..filter.gaussian import gaussian
from ..filter.pool import max_pool
from ..filter.map import mapped_imgs
from ..filter.pool import max_pool


PeakVal = namedtuple("PeakVal", ["radius", "t", "r", "v"])


def simple_peaks(img, mask, gaussian, maxpool):
    g = gaussian(img[None, ...], mask)[0]
    m = max_pool(g, [maxpool, maxpool], [1, 1], "valid")
    y, x = jnp.where(g == m)
    return np.array(y), np.array(x)


def find_peaks(imgs, mask, radius):
    gl, t, r, v = (np.array(o) for o in _find_peaks(imgs, mask, tuple(radius)))
    y, x = np.where(np.isfinite(v))
    return gl, t[y, x], y, x, r[y, x], v[y, x]


def _find_peaks(imgs, cond, radius, gxy=None):
    if gxy is None:
        nt, h, w = imgs.shape
        gxy = np.meshgrid(np.arange(w), np.arange(h))
    nt, h, w = imgs.shape
    gl = gaussian_laplace_multi(imgs, radius, -3)
    max_gl = max_pool(gl, (3, 3, 3), (1, 1, 1), "same")
    glp = jnp.where((gl == max_gl) & cond, gl, -jnp.inf).reshape(-1, h, w)
    idx = jnp.argmax(glp.reshape(-1, h, w), axis=0)
    t, r = jnp.divmod(idx, len(radius))
    gx, gy = gxy
    return gl, t, r, glp[idx, gy, gx]


def find_peaks_batch(data, radius, batch=(1, 100), pbar=None):

    def prepare(start, end):
        return (
            jnp.arange(start, end),
            jnp.array(imgs[start:end], jnp.float32),
            jnp.array(avgt[start:end], jnp.float32),
        )

    def apply(ts, imgs, avgt):
        cond = (ts >= 0)[:, None, None, None]
        if mask is not None:
            cond &= mask
        imgs = (imgs - avgx - avgt[:, None, None]) / std0
        gl, t, r, g = _find_peaks(imgs, cond, radius, (gx, gy))
        t = ts[t]
        return t, r, g

    def aggregate(t, r, v):
        idx = jnp.argmax(v, axis=0)
        t = t[idx, gy, gx]
        r = r[idx, gy, gx]
        v = v[idx, gy, gx]
        return t, r, v

    def init():
        return (
            jnp.full((h, w), -1, dtype=jnp.int32),
            jnp.full((h, w), -1, dtype=jnp.int32),
            jnp.full((h, w), jnp.nan),
        )

    def append(out, val):
        outt, outr, outv = out
        t, r, v = val
        cond = outv > v
        outt = jnp.where(cond, outt, t)
        outr = jnp.where(cond, outr, r)
        outv = jnp.where(cond, outv, v)
        return outt, outr, outv

    def finish(out):
        return PeakVal(radius, *(np.array(o) for o in out))

    imgs, mask, avgx, avgt, std0 = data
    nt, h, w = imgs.shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))

    radius = tuple(radius)
    nr = len(radius)

    if pbar is not None:
        pbar.reset(total=nt)
        pbar.set_description("find")
    return mapped_imgs(nt, prepare, apply, aggregate, init, append, finish, batch, pbar)
