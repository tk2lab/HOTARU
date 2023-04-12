from collections import namedtuple

import jax
import jax.numpy as jnp

from ..filter.laplace import gaussian_laplace_multi
from ..filter.pool import max_pool
from ..filter.map import mapped_imgs


jnp.set_printoptions(precision=3, suppress=True)

PeakVal = namedtuple("PeakVal", ["t", "r", "v"])


def find_peak(imgs, mask, radius=None, size=3):
    if radius is None:
        imgs = imgs[..., None, :, :]
        rsize = 1
    else:
        imgs = gaussian_laplace_multi(imgs, radius, -3)
        rsize = 3
    max_imgs = max_pool(imgs, (rsize, size, size), (1, 1, 1), "same")
    t, r, y, x = jnp.where((imgs == max_imgs) & mask)
    v = imgs[t, r, y, x]
    if radius is not None:
        r = jnp.array(radius)[r]
    return t, r, y, x, v


def find_peak_batch(imgs, stats, radius, pbar=None):
    nt, x0, y0, mask, avgx, avgt, std0 = stats
    h, w = mask.shape
    imgs = imgs[:, y0:y0+h, x0:x0+w]

    nr = len(radius)

    def calc(t0, imgs):
        #avgt = imgs[:, mask].mean(axis=-1)
        #imgs = (imgs - avgx - avgt[:, None, None]) / std0
        imgs = (imgs - avgx) / std0
        gl = gaussian_laplace_multi(imgs, radius, -3)
        max_gl = max_pool(gl, (3, 3, 3), (1, 1, 1), "same")
        glp = jnp.where((gl == max_gl) & mask, gl, -jnp.inf)
        v = glp.max(axis=(0, 1))
        idx = jnp.argmax(glp.reshape(-1, h, w), axis=0)
        t, r = jnp.divmod(idx, nr)
        t += t0[0]
        return t, r, v, gl, glp

    def aggregate(t, r, v, gl, glp):
        print(jnp.transpose(gl[0], [2, 3, 0, 1])[100:103, 100:103])
        print(jnp.transpose(glp[0], [2, 3, 0, 1])[100:103, 100:103])
        print(t[0, 100:103, 100:103], r[0, 100:103, 100:103], v[0, 100:103, 100:103])
        idx = jnp.argmax(v, axis=0)
        x, y = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
        t = t[idx, y, x]
        r = r[idx, y, x]
        v = v[idx, y, x]
        return t, r, v

    def finish(*args):
        return aggregate(*(jnp.stack(o) for o in args))

    scale = 1000 * nr
    t, r, v = mapped_imgs(imgs, calc, aggregate, finish, scale, pbar)
    r = jnp.array(radius)[r]
    return PeakVal(t, r, v)
