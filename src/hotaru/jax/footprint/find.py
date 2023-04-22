from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ...io.mask import mask_range
from ..filter.laplace import gaussian_laplace_multi
from ..filter.map import mapped_imgs
from ..filter.pool import max_pool


PeakVal = namedtuple("PeakVal", ["radius", "t", "r", "v"])


def find_peaks(imgs, mask, radius):
    return PeakVal(*_find_peaks(imgs, cond, radius))


def _find_peaks(imgs, cond, radius, gxy=None):
    if gxy is None:
        gxy = np.meshgrid(np.arange(w), np.arange(h))
    nt, h, w = imgs.shape
    gl = gaussian_laplace_multi(imgs, radius, -3)
    max_gl = max_pool(gl, (3, 3, 3), (1, 1, 1), "same")
    glp = jnp.where((gl == max_gl) & cond, gl, -jnp.inf).reshape(-1, h, w)
    idx = jnp.argmax(glp.reshape(-1, h, w), axis=0)
    t, r = jnp.divmod(idx, len(radius))
    gx, gy = gxy
    return t, r, glp[idx, gy, gx]


def find_peaks_batch(imgs, mask, avgx, avgt, std0, radius, batch=(1, 100), pbar=None):

    def prepare(start, end):
        return (
            jnp.arange(start, end),
            jnp.array(imgs[start:end], jnp.float32),
            jnp.array(avgt[start:end], jnp.float32),
        )

    def apply(ts, imgs, avgt):
        cond = (ts >= 0)[:, None, None, None] & mask
        imgs = (imgs - avgx - avgt[:, None, None]) / std0
        t, r, g = _find_peaks(imgs, cond, radius, (gx, gy))
        t = ts[t]
        return t, r, g

    def aggregate(t, r, v):
        idx = np.argmax(v, axis=0)
        t = t[idx, gy, gx]
        r = r[idx, gy, gx]
        v = v[idx, gy, gx]
        return t, r, v

    def finish(*args):
        t, r, v = (np.stack(o, axis=0) for o in args)
        t, r, v = aggregate(t, r, v)
        r = np.array(radius)[r]
        return PeakVal(radius, t, r, v)

    x0, y0, w, h = mask_range(mask)
    nt = imgs.shape[0]
    imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
    mask = mask[y0 : y0 + h, x0 : x0 + w]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))

    radius = tuple(radius)
    nr = len(radius)

    if pbar is not None:
        pbar = pbar(total=nt)
    return mapped_imgs(nt, prepare, apply, aggregate, finish, batch, pbar.update)
