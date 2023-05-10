from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=["r"])
def gaussian(imgs, r):
    return _gaussian(imgs, r, 4 * np.ceil(r))


def _gaussian(imgs, r, nd):
    sqrt_2pi = jnp.sqrt(2 * jnp.pi)
    d = jnp.square(jnp.arange(-nd, nd + 1, 1))
    r2 = jnp.square(r)
    o0 = jnp.exp(-d / r2 / 2) / r / sqrt_2pi
    g = imgs[..., None, :, :]
    g = lax.conv(g, o0[None, None, :, None], (1, 1), "same")
    g = lax.conv(g, o0[None, None, None, :], (1, 1), "same")
    return g[..., 0, :, :]
