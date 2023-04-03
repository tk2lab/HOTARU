import jax.lax as lax
import jax.numpy as jnp


def max_pool(inputs, window, strides, padding):
    ndiff = inputs.ndim - len(window)
    window = (1,) * ndiff + window
    strides = (1,) * ndiff + strides
    return lax.reduce_window(inputs, -jnp.inf, lax.max, window, strides, padding)
