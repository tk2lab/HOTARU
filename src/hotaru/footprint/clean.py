from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..filter.laplace import gaussian_laplace_multi
from ..filter.map import mapped_imgs
from ..io.saver import (
    load,
    save,
)
from .segment import get_segment_mask

Peaks = namedtuple("Peaks", "idx r y x radius intensity")


def clean_segment_batch(imgs, mask, radius, batch=100, pbar=None):
    def prepare(start, end):
        return jnp.array(imgs[start:end], jnp.float32)

    def apply(imgs):
        gl = gaussian_laplace_multi(imgs, radius, -3)
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

    def aggregate(*args):
        return tuple(np.concatenate(a, axis=0) for a in args)

    def append(out, data):
        out.append(data)
        return out

    def finish(out):
        args = zip(*out)
        imgs, *stats = tuple(np.concatenate(a, axis=0)[:nt] for a in args)
        t, r, y, x, intensity = stats
        return imgs, Peaks(t, r, y, x, np.array(radius)[r], intensity)

    radius = tuple(radius)
    nt, h, w = imgs.shape

    if pbar is not None:
        pbar.reset(total=nt)
    return mapped_imgs(nt, prepare, apply, aggregate, list, append, finish, batch, pbar)
