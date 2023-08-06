from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import get_progress
from ..filter.laplace import gaussian_laplace_single
from ..filter.map import mapped_imgs
from .segment import get_segment_mask


def make_segments_simple(imgs, y, x, r):
    g = gaussian_laplace_single(imgs, r)
    seg = jax.vmap(get_segment_mask)(g, y, x)
    val = jnp.where(seg, g, jnp.nan)
    dmin = jnp.nanmin(val, axis=(1, 2), keepdims=True)
    dmax = jnp.nanmax(val, axis=(1, 2), keepdims=True)
    out = jnp.where(seg, (g - dmin) / (dmax - dmin), 0)
    return out


def make_segments(data, peaks, batch=(1, 100), pbar=None):
    def prepare(start, end):
        return (
            jnp.array(imgs[tsr[start:end]], jnp.float32),
            jnp.array(avgt[tsr[start:end]], jnp.float32),
            jnp.array(ysr[start:end], jnp.int32),
            jnp.array(xsr[start:end], jnp.int32),
        )

    def _calc(imgs, avgt, y, x, r):
        if mask is not None:
            imgs.at[:, ~mask].set(jnp.nan),
        imgs = (imgs - avgx - avgt[:, jnp.newaxis, jnp.newaxis]) / std0
        return make_segments_simple(imgs, y, x, r)

    imgs, mask, _, avgx, avgt, std0, *_ = data
    ts, ys, xs, rs = (np.array(v) for v in (peaks.t, peaks.y, peaks.x, peaks.r))

    types = [("stack", jnp.float32)]
    nt, h, w = imgs.shape
    nk = rs.size

    pbar = get_progress(pbar)
    pbar.session("make")
    pbar.set_count(ts.size)

    out = np.empty((nk, h, w), np.float32)
    for r in np.unique(rs):
        idx = np.where(rs == r)[0]
        tsr, ysr, xsr = (v[idx] for v in (ts, ys, xs))
        calc = partial(_calc, r=r)
        o, = mapped_imgs(tsr.size, prepare, calc, types, batch, pbar)
        for i, oi in zip(idx, o):
            out[i] = oi
    return out
