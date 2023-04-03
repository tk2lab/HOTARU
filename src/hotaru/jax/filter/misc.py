import jax.lax as lax
import jax.numpy as jnp
from flax.linen import max_pool


__all__ = [
    "max_pool",
    "neighbor",
]


def neighbor(imgs):
    kernel = jnp.array([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], jnp.float32) / 8
    return lax.conv(imgs[..., None, :, :], kernel, (1, 1), "same")[..., 0, :, :]
