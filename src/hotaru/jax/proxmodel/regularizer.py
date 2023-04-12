import jax.numpy as jnp


class Regularizer:
    def __call__(self, x):
        return 0

    def prox(self, y, eta):
        return y


class NonNegativeL1:
    def __init__(self, l):
        self.l = l

    def __call__(self, x):
        return self.l * jnp.maximum(0, x).sum()

    def prox(self, y, eta):
        return jnp.maximum(0, y - self.l * eta)
