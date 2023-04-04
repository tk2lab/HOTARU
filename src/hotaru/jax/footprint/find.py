from collections import namedtuple

import jax
import jax.numpy as jnp

from ..filter.laplace import gaussian_laplace_multi
from ..filter.normalized import apply_to_normalized
from ..filter.pool import max_pool
from ..utils.saver import SaverMixin

global_buffer = 2**30


class PeakVal(namedtuple("PeakVal", ["val", "t", "r"]), SaverMixin):
    pass


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


def find_peak_batch(imgs, radius, stats=None, buffer=None, num_devices=None, pbar=None):
    if num_devices is None:
        num_devices = jax.local_device_count()
    if buffer is None:
        buffer = global_buffer
    nt, h, w = imgs.shape
    nr = len(radius)
    size = 4 * 4 * nr * h * w
    batch = (buffer + size - 1) // size

    def apply(imgs, mask):
        gl = gaussian_laplace_multi(imgs, radius, -3)
        max_gl = max_pool(gl, (3, 3, 3), (1, 1, 1), "same")
        return jnp.where((gl == max_gl) & mask, gl, -jnp.inf)

    def finish(glp):
        return jnp.concatenate(glp, axis=0)

    gen = apply_to_normalized(apply, finish, imgs, stats, batch, num_devices, pbar)
    val = jnp.full((h, w), -jnp.inf)
    ts = jnp.full((h, w), -1)
    rs = jnp.zeros((h, w))
    for t0, glp in gen:
        v = glp.max(axis=(0, 1))
        idx = jnp.argmax(glp.reshape(-1, h, w), axis=0)
        t, r = jnp.divmod(idx, nr)
        t += t0
        r = jnp.array(radius)[r]
        cond = v > val
        val = jnp.where(cond, v, val)
        ts = jnp.where(cond, t, ts)
        rs = jnp.where(cond, r, rs)

    return PeakVal(val, ts, rs)
