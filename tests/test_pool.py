from flax.linen import max_pool
import jax.numpy as jnp

nt, h, w, nr = 5, 10, 10, 3
data = jnp.arange(nt * h * w * nr).reshape(nt, h, w, nr)
maxp = max_pool(data, (3, 3, 3), (1, 1, 1), "same")
i, j, k, l = jnp.where(data == maxp)
print(data[i, j, k, l])
