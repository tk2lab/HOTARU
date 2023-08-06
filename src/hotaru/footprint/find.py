from collections import namedtuple

import jax.numpy as jnp
import numpy as np

from ..filter import (
    gaussian,
    gaussian_laplace,
    mapped_imgs,
    max_pool,
)
from ..utils import get_progress
from .radius import get_radius

PeakVal = namedtuple("PeakVal", ["radius", "t", "r", "v"])


def find_peaks(data, radius, batch=(1, 100), pbar=None):
    def prepare(start, end):
        return (
            jnp.arange(start, end),
            jnp.array(imgs[start:end], jnp.float32),
            jnp.array(avgt[start:end], jnp.float32),
        )

    def calc(ts, imgs, avgt):
        imgs = (imgs - avgx - avgt[:, jnp.newaxis, jnp.newaxis]) / std0
        g, t, r = _find_peaks(imgs, mask, radius)
        t = ts[t]
        return g, t, r

    types = [
        ("argmax", jnp.float32),
        ("argmax", jnp.int32),
        ("argmax", jnp.int32),
    ]

    imgs, mask, _, avgx, avgt, std0, *_ = data
    nt, h, w = imgs.shape
    radius = get_radius(**radius)

    pbar = get_progress(pbar)
    pbar.session("find")
    pbar.set_count(nt)
    out = mapped_imgs(nt, prepare, calc, types, batch, pbar)
    return PeakVal(*(np.array(o) for o in (radius,) + out))


def simple_peaks(img, gauss, maxpool, pbar=None):
    if pbar is not None:
        pbar.set_count(3)
    g = gaussian(img[None, ...], gauss)[0]
    if pbar is not None:
        pbar.update(1)
    m = max_pool(g, (maxpool, maxpool), (1, 1), "same")
    if pbar is not None:
        pbar.update(1)
    y, x = jnp.where(g == m)
    if pbar is not None:
        pbar.update(1)
    return np.array(y), np.array(x)


def simple_find(imgs, mask, radius):
    v, t, r = (np.array(o) for o in _find_peaks(imgs, mask, get_radius(**radius)))
    idx = jnp.where(np.isfinite(v))
    return t[idx], idx[:, 0], idx[:, 1], r[idx], v[idx]


def _find_peaks(imgs, mask, radius):
    nt, h, w = imgs.shape
    gl = gaussian_laplace(imgs, radius, 1)
    gl_max = max_pool(gl, (3, 3, 3), (1, 1, 1), "same")
    gl_peak = gl == gl_max
    if mask is not None:
        gl_peak &= mask
    gl_peak_val = jnp.where(gl_peak, gl, -jnp.inf)
    idx = jnp.argmax(gl_peak_val.reshape(-1, h, w), axis=0)
    t, r = jnp.divmod(idx, len(radius))
    return gl_peak_val.max(axis=(0, 1)), t, r
