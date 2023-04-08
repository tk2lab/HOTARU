from collections import namedtuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ...io.mask import mask_range
from ...utils.stats import Stats
from .map import mapped_imgs
from .misc import neighbor


ImageStats = namedtuple("ImageStats", "imin imax istd icor")


def calc_stats(imgs, mask=None, pbar=None):
    nt, h0, w0 = imgs.shape
    if mask is None:
        mask = np.ones((h0, w0), bool)
    x0, y0, w, h = mask_range(mask)
    imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
    mask = mask[y0 : y0 + h, x0 : x0 + w]
    mask = jnp.asarray(mask, bool)

    def calc(t0, imgs):
        avgt = imgs[:, mask].mean(axis=-1)
        diff = imgs - avgt[..., None, None]
        neig = neighbor(diff)
        sumi = diff.sum(axis=-3)
        sumn = neig.sum(axis=-3)
        sqi = jnp.square(diff).sum(axis=-3)
        sqn = jnp.square(neig).sum(axis=-3)
        cor = (diff * neig).sum(axis=-3)
        imin = diff.min(axis=-3)
        imax = diff.max(axis=-3)
        return avgt, sumi, sumn, sqi, sqn, cor, imin, imax

    def finish(avgt, sumi, sumn, sqi, sqn, cor, imin, imax):
        avgt = jnp.concatenate(avgt, axis=0)
        sumi = jnp.sum(sumi, axis=0)
        sumn = jnp.sum(sumn, axis=0)
        sqi = jnp.sum(sqi, axis=0)
        sqn = jnp.sum(sqn, axis=0)
        cor = jnp.sum(cor, axis=0)
        imin = jnp.min(imin, axis=0)
        imax = jnp.max(imax, axis=0)
        return avgt, sumi, sumn, sqi, sqn, cor, imin, imax

    scale = 2
    avgt, sumi, sumn, sqi, sqn, cor, imin, imax = mapped_imgs(imgs, calc, finish, scale, pbar)

    avgx = sumi / nt
    avgn = sumn / nt
    varx = sqi / nt - jnp.square(avgx)
    stdx = jnp.sqrt(varx)
    stdn = jnp.sqrt(sqn / nt - jnp.square(avgn))
    std0 = jnp.sqrt(varx[mask].mean())

    imin = (imin - avgx) / std0
    imax = (imax - avgx) / std0
    istd = stdx / std0
    icor = (cor / nt - avgx * avgn) / (stdx * stdn)

    imin.at[~mask].set(0)
    imax.at[~mask].set(0)
    istd.at[~mask].set(0)
    icor.at[~mask].set(0)

    stats = Stats(nt, y0, y0, mask, avgx, avgt, std0)
    istats = ImageStats(imin, imax, istd, icor)
    return stats, istats
