from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..filter.laplace import gaussian_laplace_multi
from ..filter.map import mapped_imgs
from ..filter.pool import max_pool

PeakVal = namedtuple("PeakVal", ["t", "y", "x", "r", "v"])


def find_peaks(imgs, mask, radius):
    imgs, mask = _find_peaks(imgs, mask, tuple(radius), len(radius))
    return imgs, *jnp.where(mask)


@partial(jax.jit, static_argnames=["radius", "rsize"])
def _find_peaks(imgs, mask, radius, rsize):
    imgs = gaussian_laplace_multi(imgs, radius)
    max_imgs = max_pool(imgs, (3, 3, 3), (1, 1, 1), "same")
    return imgs, (imgs == max_imgs) & mask[..., None]


def find_peaks_batch(imgs, stats, radius, pbar=None):
    nt, x0, y0, mask, avgx, avgt, std0 = stats
    h, w = mask.shape
    gx, gy = jnp.meshgrid(jnp.arange(w), jnp.arange(h))

    imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
    radius = tuple(radius)
    nr = len(radius)

    def calc(ts, imgs):
        cond = (ts >= 0)[:, None, None, None]
        avgt = imgs[..., mask].mean(axis=-1)[..., None, None]
        imgs = (imgs - avgx - avgt) / std0
        gl = gaussian_laplace_multi(imgs, radius, -3)
        max_gl = max_pool(gl, (3, 3, 3), (1, 1, 1), "same")
        glp = jnp.where((gl == max_gl) & cond & mask, gl, -jnp.inf).reshape(-1, h, w)
        idx = jnp.argmax(glp.reshape(-1, h, w), axis=0)
        t, r = jnp.divmod(idx, nr)
        t = ts[t]
        return t, r, glp[idx, gy, gx]

    def aggregate(t, r, v):
        idx = np.argmax(v, axis=0)
        t = t[idx, gy, gx]
        r = r[idx, gy, gx]
        v = v[idx, gy, gx]
        y, x = np.where(v > -jnp.inf)
        t = t[y, x]
        r = r[y, x]
        v = v[y, x]
        return t, y, x, r, v

    def finish(*args):
        t, y, x, r, v = (np.concatenate(o, axis=0) for o in args)
        idx = np.argsort(v)[::-1]
        r = jnp.array(radius)[r]
        return PeakVal(*(o[idx] for o in (t, y, x, r, v)))

    scale = 50 * nr
    return mapped_imgs(imgs, calc, aggregate, finish, scale, pbar)
