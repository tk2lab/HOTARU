from functools import partial

import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp


def gaussian_laplace(imgs, r):
    return _gaussian_laplace(imgs, r, 4 * np.ceil(r))


def gaussian_laplace_multi(imgs, rs, axis=-1):
    return jnp.stack([gaussian_laplace(imgs, r) for r in rs], axis=axis)


@partial(jax.jit, static_argnames=["nd"])
def _gaussian_laplace(imgs, r, nd):
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
