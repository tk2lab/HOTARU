import tqdm
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp

from ..io.mask import mask_range


global_buffer = 2 ** 30


def gaussian_laplace(imgs, r, nd):
    sqrt_2pi = jnp.sqrt(2 * jnp.pi)
    mr = jnp.ceil(r).astype(jnp.int32)
    d = jnp.square(jnp.arange(-nd, nd + 1, dtype=jnp.float32))
    r2 = jnp.square(r)
    o0 = jnp.exp(-d / r2 / 2) / r / sqrt_2pi
    o2 = (1 - d / r2) * o0
    gl1 = lax.conv(imgs[..., None, :, :], o2[None, None, :, None], (1, 1), "same")
    gl1 = lax.conv(gl1, o0[None, None, None, :], (1, 1), "same")
    gl2 = lax.conv(imgs[..., None, :, :], o2[None, None, None, :], (1, 1), "same")
    gl2 = lax.conv(gl2, o0[None, None, :, None], (1, 1), "same")
    return (gl1 + gl2)[..., 0, :, :]


def gaussian_laplace_multi(imgs, radius, avgt=None, avgx=None, mask=None, nd=None, buffer=None):
    nt, h0, w0 = imgs.shape
    nr = len(radius)

    if mask is None:
        mask = np.ones((h0, w0), bool)
    x0, y0, w, h = mask_range(mask)
    imgs = imgs[:, y0:y0+h, x0:x0+w]
    mask = mask[y0:y0+h, x0:x0+w]

    if avgt is None:
        avgt = jnp.zeros(nt)
    if avgx is None:
        avgx = jnp.zeros((h, w))
    jmask = jnp.asarray(mask, bool)

    if nd is None:
        nd = jax.local_device_count()
    if buffer is None:
        buffer = global_buffer
    size = 4 * nd * h * w
    batch = (buffer + size - 1) // size

    calc = jax.pmap(gaussian_laplace, static_broadcasted_argnums=[1, 2])

    out = jnp.empty((nt, nr, h, w))
    with tqdm.tqdm(total=nt) as pbar:
        while pbar.n < nt:
            start = pbar.n
            end = min(nt, start + nd * batch)
            if end == nt:
                nd = 1
                batch = end - start
            clip = jnp.asarray(imgs[start:end], jnp.float32)
            clip = (clip - avgx - avgt[start:end, None, None]).reshape(nd, batch, h, w)
            for i, r in enumerate(radius):
                tmp = jnp.concatenate(calc(clip, r, 4 * np.ceil(r)), axis=0)
                out = out.at[start:end, i, ...].set(tmp)
            pbar.update(end - start)
    return out
