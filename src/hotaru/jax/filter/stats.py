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


def calc_stats(imgs, mask=None, nd=None, buffer=None):
    nt, h0, w0 = imgs.shape
    if mask is None:
        mask = np.ones((h0, w0), bool)
    x0, y0, w, h = mask_range(mask)
    imgs = imgs[:, y0:y0+h, x0:x0+w]
    mask = mask[y0:y0+h, x0:x0+w]
    jmask = jnp.asarray(mask, bool)

    if nd is None:
        nd = jax.local_device_count()
    if buffer is None:
        buffer = global_buffer
    size = 4 * nd * h * w
    batch = (buffer + size - 1) // size

    avgt = jnp.zeros((nt,))
    sumi = jnp.zeros((h, w))
    sumn = jnp.zeros((h, w))
    sqi = jnp.zeros((h, w))
    sqn = jnp.zeros((h, w))
    cor = jnp.zeros((h, w))
    maxi = jnp.full((h, w), -np.inf)

    @jax.pmap
    def calc(imgs):
        avgt = imgs[:, jmask].mean(axis=-1)
        diff = imgs - avgt[..., None, None]
        neig = neighbor(diff)
        sumi = diff.sum(axis=-3)
        sumn = neig.sum(axis=-3)
        sqi = jnp.square(diff).sum(axis=-3)
        sqn = jnp.square(neig).sum(axis=-3)
        cor = (diff * neig).sum(axis=-3)
        maxi = diff.max(axis=-3)
        return avgt, sumi, sumn, sqi, sqn, cor, maxi

    with tqdm.tqdm(total=nt) as pbar:
        while pbar.n < nt:
            start = pbar.n
            end = min(nt, start + nd * batch)
            if end == nt:
                nd = 1
                batch = end - start
            clip = jnp.asarray(imgs[start:end], jnp.float32).reshape(nd, batch, h, w)
            _avgt, _sumi, _sumn, _sqi, _sqn, _cor, _maxi = calc(clip)
            avgt = avgt.at[start:end].set(jnp.concatenate(_avgt, axis=0))
            sumi += _sumi.sum(axis=0)
            sumn += _sumn.sum(axis=0)
            sqi += _sqi.sum(axis=0)
            sqn += _sqn.sum(axis=0)
            cor += _cor.sum(axis=0)
            maxi = jnp.maximum(maxi, _maxi.max(axis=0))
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

    avgx.at[~mask].set(0)
    maxi.at[~mask].set(0)
    stdi.at[~mask].set(0)
    cori.at[~mask].set(0)
    return avgt, avgx, std0, maxi, stdi, cori
