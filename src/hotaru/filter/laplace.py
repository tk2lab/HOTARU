from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=["rs", "axis"])
def gaussian_laplace(imgs, rs, axis=-1):
    ndim = imgs.ndim
    if axis < 0:
        axis = ndim + axis + 1
    out_shape = imgs.shape[:axis] + (len(rs),) + imgs.shape[axis:]
    out = jnp.empty(out_shape)
    for i, r in enumerate(rs):
        idx = (slice(None),) * axis + (i,) + (slice(None),) * (ndim - axis)
        out = out.at[idx].set(gaussian_laplace_single(imgs, r))
    return out


@partial(jax.jit, static_argnames=["r"])
def gaussian_laplace_single(imgs, r):
    return _gaussian_laplace(imgs, r, 4 * np.ceil(r))


def _gaussian_laplace(imgs, r, nd):
    sqrt_2pi = jnp.sqrt(2 * jnp.pi)
    d = jnp.square(jnp.arange(-nd, nd + 1, 1))
    r2 = jnp.square(r)
    o0 = jnp.exp(-d / r2 / 2) / r / sqrt_2pi
    o2 = (1 - d / r2) * o0
    o0 = jnp.expand_dims(o0, (0, 1))
    o2 = jnp.expand_dims(o2, (0, 1))
    gl1 = lax.conv(jnp.expand_dims(imgs, -3), jnp.expand_dims(o2, -1), (1, 1), "same")
    gl1 = lax.conv(gl1, jnp.expand_dims(o0, -2), (1, 1), "same")
    gl2 = lax.conv(jnp.expand_dims(imgs, -3), jnp.expand_dims(o2, -2), (1, 1), "same")
    gl2 = lax.conv(gl2, jnp.expand_dims(o0, -1), (1, 1), "same")
    return (gl1 + gl2)[..., 0, :, :]
