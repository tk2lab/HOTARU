from collections import namedtuple

import jax.numpy as jnp
import numpy as np

from .map import mapped_imgs
from .neighbor import neighbor

Stats = namedtuple("Stats", "avgx avgt std0 imin imax istd icor")


def calc_stats(imgs, mask=None, batch=(1, 100), pbar=None):
    def prepare(start, end):
        return jnp.array(imgs[start:end], jnp.float32)

    def calc(imgs):
        if mask is None:
            masked = imgs.reshape(-1, h * w)
        else:
            masked = imgs[:, mask]
        avgt = jnp.nanmean(masked, axis=-1)
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

    def init():
        avgt = []
        sumi = jnp.zeros((h, w))
        sumn = jnp.zeros((h, w))
        sqi = jnp.zeros((h, w))
        sqn = jnp.zeros((h, w))
        cor = jnp.zeros((h, w))
        imin = jnp.zeros((h, w))
        imax = jnp.zeros((h, w))
        return [avgt, sumi, sumn, sqi, sqn, cor, imin, imax]

    def append(out, val):
        out[0].append(val[0])
        out[1] += val[1]
        out[2] += val[2]
        out[3] += val[3]
        out[4] += val[4]
        out[5] += val[5]
        out[6] = np.maximum(out[6], val[6])
        out[7] = np.minimum(out[7], val[7])
        return out

    def finish(out):
        avgt, sumi, sumn, sqi, sqn, cor, imin, imax = out
        avgt = jnp.concatenate(avgt, axis=0)[:nt]
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

        return Stats(*map(np.array, (avgx, avgt, std0, imin, imax, istd, icor)))

    nt, h, w = imgs.shape

    if pbar is not None:
        pbar.set_count(nt)
    return mapped_imgs(nt, prepare, calc, aggregate, init, append, finish, batch, pbar)
