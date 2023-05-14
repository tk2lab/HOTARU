from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..io.saver import (
    load,
    save,
)
from ..filter.laplace import gaussian_laplace
from ..filter.map import mapped_imgs
from .segment import get_segment_mask


def make_segment(imgs, y, x, r):
    g = gaussian_laplace(imgs, r)
    seg = jax.vmap(get_segment_mask)(g, y, x)
    dmin = jnp.nanmin(jnp.where(seg, g, jnp.nan), axis=(1, 2), keepdims=True)
    dmax = jnp.nanmax(jnp.where(seg, g, jnp.nan), axis=(1, 2), keepdims=True)
    return jnp.where(seg, (g - dmin) / (dmax - dmin), 0)


def make_segment_batch(data, peaks, batch=100, pbar=None):

    def prepare(start, end):
        return (
            jnp.array(imgs[tsr[start:end]], jnp.float32),
            jnp.array(avgt[tsr[start:end]], jnp.float32),
            jnp.array(ysr[start:end], jnp.int32),
            jnp.array(xsr[start:end], jnp.int32),
        )

    def _apply(imgs, avgt, y, x, r):
        if mask is not None:
            imgs.at[:, ~mask].set(jnp.nan),
        imgs = (imgs - avgx - avgt[:, None, None]) / std0
        return make_segment(imgs, y, x, r)

    def aggregate(seg):
        return jnp.concatenate(seg, axis=0)

    def append(out, data):
        out.append(data)
        return out

    def finish(seg):
        return np.array(aggregate(seg))

    imgs, mask, avgx, avgt, std0 = data
    nt, h, w = imgs.shape

    ts, ys, xs, rs = (np.array(v) for v in (peaks.t, peaks.y, peaks.x, peaks.r))
    nk = rs.size
    out = np.empty((nk, h, w), np.float32)
    if pbar is not None:
        pbar.reset(total=ts.size)
        pbar.set_description("make")
    for r in np.unique(rs):
        idx = np.where(rs == r)[0]
        tsr, ysr, xsr = (v[idx] for v in (ts, ys, xs))
        apply = partial(_apply, r=r)
        o = mapped_imgs(tsr.size, prepare, apply, finish, list, append, finish, batch, pbar)
        for i, oi in zip(idx, o):
            out[i] = oi
    if mask is None:
        out = out.reshape(-1, h * w)
    else:
        out = out[:, mask]
    return out
