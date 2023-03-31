from collections import namedtuple

import tqdm
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp

from ..io.mask import mask_range


global_buffer = 2 ** 30


def neighbor(imgs):
    kernel = jnp.array([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], jnp.float32) / 8
    return lax.conv(imgs[..., None, :, :], kernel, (1, 1), "same")[..., 0, :, :]


class Stats(namedtuple("Stats", ["nt", "x0", "y0", "mask", "avgx", "avgt", "std0"])):

    def save(self, path):
        jnp.savez(path, **self._asdict())

    @classmethod
    def load(cls, path):
        with jnp.load(path) as npz:
            stats = cls(*[npz[n] for n in cls._fields])
        return stats


def calc_stats(imgs, mask=None, buffer=None, num_devices=None):
    nt, h0, w0 = imgs.shape
    if mask is None:
        mask = np.ones((h0, w0), bool)
    x0, y0, w, h = mask_range(mask)
    imgs = imgs[:, y0:y0+h, x0:x0+w]
    mask = mask[y0:y0+h, x0:x0+w]
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
        maxi = diff.max(axis=-3)
        return avgt, sumi, sumn, sqi, sqn, cor, maxi

    def pmap_calc(imgs):
        imgs = imgs.reshape(num_devices, batch, h, w)
        avgt, sumi, sumn, sqi, sqn, cor, maxi = jax.pmap(calc)(imgs)
        avgt = jnp.concatenate(avgt, axis=0)
        sumi = sumi.sum(axis=0)
        sumn = sumn.sum(axis=0)
        sqi = sqi.sum(axis=0)
        sqn = sqn.sum(axis=0)
        cor = cor.sum(axis=0)
        maxi = maxi.max(axis=0)
        return avgt, sumi, sumn, sqi, sqn, cor, maxi

    avgt = jnp.zeros((nt,))
    sumi = jnp.zeros((h, w))
    sumn = jnp.zeros((h, w))
    sqi = jnp.zeros((h, w))
    sqn = jnp.zeros((h, w))
    cor = jnp.zeros((h, w))
    maxi = jnp.full((h, w), -np.inf)

    with tqdm.tqdm(total=nt) as pbar:
        while pbar.n < nt:
            start = pbar.n
            end = min(nt, start + num_devices * batch)
            batch = end - start
            if end == nt:
                num_devices = 1
            clip = jnp.asarray(imgs[start:end], jnp.float32)
            _avgt, _sumi, _sumn, _sqi, _sqn, _cor, _maxi = pmap_calc(clip)
            avgt = avgt.at[start:end].set(_avgt)
            sumi += _sumi
            sumn += _sumn
            sqi += _sqi
            sqn += _sqn
            cor += _cor
            maxi = jnp.maximum(maxi, _maxi)
            pbar.update(end - start)

    avgx = sumi / nt
    avgn = sumn / nt
    varx = sqi / nt - jnp.square(avgx)
    stdx = jnp.sqrt(varx)
    stdn = jnp.sqrt(sqn / nt - jnp.square(avgn))
    std0 = jnp.sqrt(varx[mask].mean())

    maxi = (maxi - avgx) / std0
    stdi = stdx / std0
    cori = (cor / nt - avgx * avgn) / (stdx * stdn)

    maxi.at[~mask].set(0)
    stdi.at[~mask].set(0)
    cori.at[~mask].set(0)

    return Stats(nt, x0, y0, mask, avgx, avgt, std0), maxi, stdi, cori
