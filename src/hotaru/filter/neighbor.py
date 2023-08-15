import jax.lax as lax
import jax.numpy as jnp

__all__ = [
    "neighbor",
]


def neighbor(imgs):
    kernel = jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], jnp.float32) / 8
    kernel = jnp.expand_dims(kernel, (0, 1))
    return lax.conv(jnp.expand_dims(imgs, -3), kernel, (1, 1), "same")[..., 0, :, :]
