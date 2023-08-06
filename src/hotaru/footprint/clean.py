from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

from ..filter import (
    gaussian_laplace,
    mapped_imgs,
)
from ..utils import get_progress
from .radius import get_radius
from .segment import get_segment_mask

Footprint = namedtuple("Footprint", "foootprit idx r y x radius intensity")


def clean_segments(vals, shape, mask, radius, batch=(1, 100), pbar=None):
    def prepare(start, end):
        return jnp.array(vals[start:end], jnp.float32)

    def calc(vals):
        if mask is None:
            imgs = vals.reshape(batch[1], *shape)
        else:
            imgs = jnp.zeros((batch[1], *shape), vals.dtype)
            imgs[:, mask] = vals
        gl = gaussian_laplace(imgs, radius, -3)
        if mask is not None:
            gl.at[:, :, ~mask].set(jnp.nan)
        nt, nr, h, w = gl.shape
        idx = jnp.argmax(gl.reshape(nt, nr * h * w), axis=1)
        t = jnp.arange(nt)
        r, y, x = idx // (h * w), (idx // w) % h, idx % w
        g = gl[t, r, y, x]
        seg = jax.vmap(get_segment_mask)(gl[t, r], y, x)
        imgs = jnp.where(seg, imgs, jnp.nan)
        dmin = jnp.nanmin(jnp.where(seg, imgs, jnp.nan), axis=(1, 2), keepdims=True)
        dmax = jnp.nanmax(jnp.where(seg, imgs, jnp.nan), axis=(1, 2), keepdims=True)
        imgs = jnp.where(seg, (imgs - dmin) / (dmax - dmin), 0)
        return imgs, t, r, y, x, g

    types = [
        ("stack", jnp.float32),
        ("stack", jnp.int32),
        ("stack", jnp.int32),
        ("stack", jnp.int32),
        ("stack", jnp.int32),
        ("stack", jnp.float32),
    ]

    radius = get_radius(**radius)
    nt = vals.shape[0]

    pbar = get_progress(pbar)
    pbar.session("clean")
    pbar.set_count(nt)
    imgs, t, r, y, x, intensity = mapped_imgs(nt, prepare, calc, types, batch, pbar)
    return Footprint(imgs, t, r, y, x, np.array(radius)[r], intensity)
