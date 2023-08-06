from collections import namedtuple

import jax.numpy as jnp
import numpy as np

from ..utils import get_progress
from .map import mapped_imgs
from .neighbor import neighbor

Stats = namedtuple("Stats", "avgx avgt std0 min0 max0 min1 max1 imax istd icor")


def calc_stats(imgs, mask=None, batch=(1, 100), pbar=None):

    def prepare(start, end):
        return jnp.array(imgs[start:end], jnp.float32)

    def calc(imgs):
        if mask is None:
            masked = imgs.reshape(-1, h * w)
        else:
            masked = imgs[:, mask]
        min0 = jnp.nanmin(imgs, axis=0)
        max0 = jnp.nanmax(imgs, axis=0)
        avgt = jnp.nanmean(masked, axis=1)
        diff = imgs - avgt[..., None, None]
        imin = jnp.nanmin(diff, axis=0)
        imax = jnp.nanmax(diff, axis=0)
        neig = neighbor(diff)
        sumi = jnp.nansum(diff, axis=0)
        sumn = jnp.nansum(neig, axis=0)
        sqi = jnp.nansum(jnp.square(diff), axis=0)
        sqn = jnp.nansum(jnp.square(neig), axis=0)
        cor = jnp.nansum(diff * neig, axis=0)
        return min0, max0, imin, imax, avgt, sumi, sumn, sqi, sqn, cor

    def finish(min0, max0, imin, imax, avgt, sumi, sumn, sqi, sqn, cor):
        avgx = sumi / nt
        avgn = sumn / nt
        varx = sqi / nt - jnp.square(avgx)
        stdx = jnp.sqrt(varx)
        stdn = jnp.sqrt(sqn / nt - jnp.square(avgn))
        if mask is None:
            varx_masked = varx.ravel()
        else:
            varx_masked = varx[mask]
        std0 = jnp.sqrt(varx_masked.mean())

        imin = (imin - avgx) / std0
        imax = (imax - avgx) / std0
        istd = stdx / std0
        icor = (cor / nt - avgx * avgn) / (stdx * stdn)

        if mask is not None:
            avgx.at[~mask].set(jnp.nan)
            imin.at[~mask].set(jnp.nan)
            imax.at[~mask].set(jnp.nan)
            istd.at[~mask].set(jnp.nan)
            icor.at[~mask].set(jnp.nan)

        stats = avgx, avgt, std0, min0.min(), max0.max(), imin.min(), imax.max()
        simgs = imax, istd, icor
        stats = Stats(*map(np.array, stats + simgs))
        return stats

    types = [
        ("min", jnp.int32),
        ("max", jnp.int32),
        ("min", jnp.float32),
        ("max", jnp.float32),
        ("stack", jnp.float32),
        ("add", jnp.float32),
        ("add", jnp.float32),
        ("add", jnp.float32),
        ("add", jnp.float32),
        ("add", jnp.float32),
    ]
    nt, h, w = imgs.shape

    pbar = get_progress(pbar)
    pbar.session("stats")
    pbar.set_count(nt)
    out = mapped_imgs(nt, prepare, calc, types, batch, pbar)
    return finish(*out)
