from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ...io.saver import (
    load,
    save,
)
from ..filter.laplace import gaussian_laplace
from ..filter.map import mapped_imgs
from .segment import get_segment_mask


def make_segment(imgs, y, x, r):
    g = gaussian_laplace(imgs, r)
    seg = jax.vmap(get_segment_mask)(g, y, x)
    dmin = jnp.nanmin(jnp.where(seg, g, jnp.nan))
    dmax = jnp.nanmax(jnp.where(seg, g, jnp.nan))
    return jnp.where(seg, (g - dmin) / (dmax - dmin), 0)


def make_segment_batch(data, peaks, batch=100, pbar=None):

    def prepare(start, end):
        return (
            jnp.array(imgs[tsr[start:end]], jnp.float32).at[:, ~mask].set(jnp.nan),
            jnp.array(avgt[tsr[start:end]], jnp.float32),
            jnp.array(ysr[start:end], jnp.int32),
            jnp.array(xsr[start:end], jnp.int32),
        )

    def _apply(imgs, avgt, y, x, r):
        imgs = (imgs - avgx - avgt[:, None, None]) / std0
        return make_segment(imgs, y, x, r)

    def finish(seg):
        return np.concatenate(seg, axis=0)

    imgs, mask, avgx, avgt, std0 = data
    nt, h, w = imgs.shape

    ts, ys, xs, rs = (np.array(v) for v in (peaks.t, peaks.y, peaks.x, peaks.r))
    nk = rs.size
    out = np.empty((nk, h, w), np.float32)
    _pbar = pbar
    for r in np.unique(rs):
        idx = np.where(rs == r)[0]
        tsr, ysr, xsr = (v[idx] for v in (ts, ys, xs))
        apply = partial(_apply, r=r)
        if _pbar is not None:
            pbar = _pbar(total=tsr.size)
            pbar.set_description(f"make {r:.3f}")
            pbar = pbar.update
        o = mapped_imgs(tsr.size, prepare, apply, finish, finish, batch, pbar)
        for i, oi in zip(idx, o):
            out[i] = oi
    return out[:, mask]
