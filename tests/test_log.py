import jax
import jax.numpy as jnp
import numpy as np

from hotaru.jax.filter.laplace import (
    gaussian_laplace,
    gaussian_laplace_multi,
    gen_gaussian_laplace,
)


nt, h, w = 1000, 100, 100
data = jnp.ones((nt, h, w))
stats = nt, 0, 0, jnp.ones((h, w), bool), jnp.zeros((h, w)), jnp.zeros(nt), jnp.ones(())
radius = [3.0, 4.0, 5.0]


for out in gen_gaussian_laplace(data, radius, stats):
    print(out)
