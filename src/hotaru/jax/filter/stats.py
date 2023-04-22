from collections import namedtuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ...io.mask import mask_range
from .map import mapped_imgs
from .misc import neighbor


Stats = namedtuple("Stats", "avgx avgt std0 imin imax istd icor")


def calc_stats(imgs, mask=None, batch=(1, 100), pbar=None):

    def prepare(start, end):
        return jnp.array(imgs[start:end], jnp.float32)

    def calc(imgs):
        avgt = jnp.nanmean(imgs[:, mask], axis=-1)
        diff = imgs - avgt[..., None, None]
        neig = neighbor(diff)
        sumi = jnp.nansum(diff, axis=-3)
        sumn = jnp.nansum(neig, axis=-3)
        sqi = jnp.nansum(jnp.square(diff), axis=-3)
        sqn = jnp.nansum(jnp.square(neig), axis=-3)
        cor = jnp.nansum(diff * neig, axis=-3)
        imin = jnp.nanmin(diff, axis=-3)
        imax = jnp.nanmax(diff, axis=-3)
        return avgt, sumi, sumn, sqi, sqn, cor, imin, imax

    def aggregate(avgt, sumi, sumn, sqi, sqn, cor, imin, imax):
        avgt = avgt.ravel()
        sumi = sumi.sum(axis=0)
        sumn = sumn.sum(axis=0)
        sqi = sqi.sum(axis=0)
        sqn = sqn.sum(axis=0)
        cor = cor.sum(axis=0)
        imin = imin.min(axis=0)
        imax = imax.max(axis=0)
        return avgt, sumi, sumn, sqi, sqn, cor, imin, imax

    def finish(avgt, *args):
        avgt = np.concatenate(avgt, axis=0)[:nt]
        iout = map(np.stack, args)
        avgt, sumi, sumn, sqi, sqn, cor, imin, imax = aggregate(avgt, *iout)

        avgx = sumi / nt
        avgn = sumn / nt
        varx = sqi / nt - np.square(avgx)
        stdx = np.sqrt(varx)
        stdn = np.sqrt(sqn / nt - np.square(avgn))
        std0 = np.sqrt(varx[mask].mean())

        imin = (imin - avgx) / std0
        imax = (imax - avgx) / std0
        istd = stdx / std0
        icor = (cor / nt - avgx * avgn) / (stdx * stdn)

        avgx[~mask] = np.nan
        imin[~mask] = np.nan
        imax[~mask] = np.nan
        istd[~mask] = np.nan
        icor[~mask] = np.nan

        return Stats(avgx, avgt, std0, imin, imax, istd, icor)

    nt, h0, w0 = imgs.shape
    if mask is None:
        mask = np.ones((h0, w0), bool)
    x0, y0, w, h = mask_range(mask)
    imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
    mask = mask[y0 : y0 + h, x0 : x0 + w]

    if pbar is not None:
        pbar = pbar(total=nt)
    return mapped_imgs(nt, prepare, calc, aggregate, finish, batch, pbar.update)
