from collections import namedtuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ..io.mask import mask_range
from ..utils.saver import SaverMixin
from .misc import neighbor

global_buffer = 2**30


class Stats(
    namedtuple(
        "Stats", ["nt", "x0", "y0", "mask", "avgx", "avgt", "std", "min", "max"]
    ),
    SaverMixin,
):
    pass


class ImageStats(
    namedtuple("ImageStats", ["max", "std", "cor"]),
    SaverMixin,
):
    pass


def calc_stats(imgs, mask=None, buffer=None, num_devices=None, pbar=None):
    nt, h0, w0 = imgs.shape
    if mask is None:
        mask = np.ones((h0, w0), bool)
    x0, y0, w, h = mask_range(mask)
    imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
    mask = mask[y0 : y0 + h, x0 : x0 + w]
    mask = jnp.asarray(mask, bool)

    if num_devices is None:
        num_devices = jax.local_device_count()
    if buffer is None:
        buffer = global_buffer
    size = 4 * h * w
    batch = (buffer + size - 1) // size

    def calc(imgs):
        avgt = imgs[:, mask].mean(axis=-1)
        diff = imgs - avgt[..., None, None]
        neig = neighbor(diff)
        sumi = diff.sum(axis=-3)
        sumn = neig.sum(axis=-3)
        sqi = jnp.square(diff).sum(axis=-3)
        sqn = jnp.square(neig).sum(axis=-3)
        cor = (diff * neig).sum(axis=-3)
        mini = diff.min(axis=-3)
        maxi = diff.max(axis=-3)
        return avgt, sumi, sumn, sqi, sqn, cor, mini, maxi

    def pmap_calc(imgs):
        imgs = imgs.reshape(num_devices, batch, h, w)
        avgt, sumi, sumn, sqi, sqn, cor, mini, maxi = jax.pmap(calc)(imgs)
        avgt = jnp.concatenate(avgt, axis=0)
        sumi = sumi.sum(axis=0)
        sumn = sumn.sum(axis=0)
        sqi = sqi.sum(axis=0)
        sqn = sqn.sum(axis=0)
        cor = cor.sum(axis=0)
        mini = mini.min(axis=0)
        maxi = maxi.max(axis=0)
        return avgt, sumi, sumn, sqi, sqn, cor, mini, maxi

    avgt = jnp.zeros((nt,))
    sumi = jnp.zeros((h, w))
    sumn = jnp.zeros((h, w))
    sqi = jnp.zeros((h, w))
    sqn = jnp.zeros((h, w))
    cor = jnp.zeros((h, w))
    mini = jnp.full((h, w), np.inf)
    maxi = jnp.full((h, w), -np.inf)

    end = 0
    while end < nt:
        start = end
        end = min(nt, start + num_devices * batch)
        batch = end - start
        if end == nt:
            num_devices = 1
        clip = jnp.asarray(imgs[start:end], jnp.float32)
        _avgt, _sumi, _sumn, _sqi, _sqn, _cor, _mini, _maxi = pmap_calc(clip)
        avgt = avgt.at[start:end].set(_avgt)
        sumi += _sumi
        sumn += _sumn
        sqi += _sqi
        sqn += _sqn
        cor += _cor
        mini = jnp.minimum(mini, _mini)
        maxi = jnp.maximum(maxi, _maxi)
        if pbar is not None:
            pbar.update(end - start)

    avgx = sumi / nt
    avgn = sumn / nt
    varx = sqi / nt - jnp.square(avgx)
    stdx = jnp.sqrt(varx)
    stdn = jnp.sqrt(sqn / nt - jnp.square(avgn))
    std0 = jnp.sqrt(varx[mask].mean())

    mini = (mini - avgx) / std0
    maxi = (maxi - avgx) / std0
    stdi = stdx / std0
    cori = (cor / nt - avgx * avgn) / (stdx * stdn)

    mini.at[~mask].set(0)
    maxi.at[~mask].set(0)
    stdi.at[~mask].set(0)
    cori.at[~mask].set(0)

    stats = Stats(nt, x0, y0, mask, avgx, avgt, std0, mini.min(), maxi.max())
    istats = ImageStats(maxi, stdi, cori)
    return stats, istats
