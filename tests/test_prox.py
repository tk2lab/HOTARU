import jax
import jax.numpy as jnp

from hotaru.jax.proxmodel.optimizer import ProxOptimizer
from hotaru.jax.proxmodel.regularizer import Regularizer, NonNegativeL1


def loss(x, y, z):
    return jnp.square(x - 2).sum() + y.sum() + z.sum(), 10 * x

reguralizers = [
    NonNegativeL1(0.1),
    NonNegativeL1(0.1),
    NonNegativeL1(0.1),
]

x0 = [
    jnp.full((100,), 1.0),
    jnp.full((100,), 1.0),
    jnp.full((100,), 1.0),
]

optimizer = ProxOptimizer(loss, reguralizers, 0.01)
optimizer.init(x0)
for i in range(1000):
    optimizer.step()
    if i % 100 == 0:
        print(optimizer.x)
        print(optimizer.loss())
