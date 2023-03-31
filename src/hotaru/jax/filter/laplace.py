from functools import partial

import tqdm
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp

from ..io.mask import mask_range


global_buffer = 2 ** 30


@partial(jax.jit, static_argnames=["nd"])
def gaussian_laplace(imgs, r, nd):
    sqrt_2pi = jnp.sqrt(2 * jnp.pi)
    d = jnp.square(jnp.arange(-nd, nd + 1, 1))
    r2 = jnp.square(r)
    o0 = jnp.exp(-d / r2 / 2) / r / sqrt_2pi
    o2 = (1 - d / r2) * o0
    gl1 = lax.conv(imgs[..., None, :, :], o2[None, None, :, None], (1, 1), "same")
    gl1 = lax.conv(gl1, o0[None, None, None, :], (1, 1), "same")
    gl2 = lax.conv(imgs[..., None, :, :], o2[None, None, None, :], (1, 1), "same")
    gl2 = lax.conv(gl2, o0[None, None, :, None], (1, 1), "same")
    return (gl1 + gl2)[..., 0, :, :]


def gaussian_laplace_multi(imgs, rs):
    return jnp.stack(
        [gaussian_laplace(imgs, r, 4 * np.ceil(r)) for r in rs],
        axis=-1,
    )


def gen_gaussian_laplace(imgs, radius, stats=None, opt_fn=None, buffer=None, num_devices=None):
    if stats is None:
        nt, h, w = imgs.shape
        x0, y0 = 0, 0
        mask = np.ones((h, w), bool)
        avgx = jnp.zeros((h, w))
        avgt = jnp.zeros(nt)
        std0 = jnp.ones(())
    else:
        nt, x0, y0, mask, avgx, avgt, std0 = stats
        h, w = mask.shape

    imgs = imgs[:, y0:y0+h, x0:x0+w]
    nr = len(radius)

    if num_devices is None:
        num_devices = jax.local_device_count()
    if buffer is None:
        buffer = global_buffer
    size = 4 * 4 * nr * h * w
    batch = (buffer + size - 1) // size

    def calc(imgs, avgt):
        imgs = (imgs - avgx - avgt) / std0
        return gaussian_laplace_multi(imgs, radius)

    def pmap_calc(imgs, avgt):
        imgs = imgs.reshape(num_devices, batch, h, w)
        avgt = avgt.reshape(num_devices, batch, 1, 1)
        out = jax.pmap(calc)(imgs, avgt)
        return jnp.concatenate(out, axis=0)

    with tqdm.tqdm(total=nt) as pbar:
        while pbar.n < nt:
            start = pbar.n
            end = min(nt, start + num_devices * batch)
            batch = end - start
            if end == nt:
                num_devices = 1
            clip = jnp.asarray(imgs[start:end], jnp.float32)
            clipt = avgt[start:end]
            yield start, pmap_calc(clip, clipt).at[:, mask, :].set(0)
            pbar.update(end - start)
