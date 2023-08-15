import jax.numpy as jnp


class Regularizer:
    def __call__(self, x):
        return 0

    def prox(self, y, eta):
        return y


class NonNegativeL1:
    def __init__(self, fac):
        self.fac = fac

    def __call__(self, x):
        #x = jnp.maximum(0, x)
        return self.fac * x.sum()

    def prox(self, y, eta):
        return jnp.maximum(0, y - self.fac * eta)


class MaxNormNonNegativeL1:
    def __init__(self, fac):
        self.fac = fac

    def __call__(self, x):
        s = x.sum(axis=-1)
        m = x.max(axis=-1)
        if x.ndim > 1:
            m = jnp.where(m > 0, m, jnp.ones(()))
            return self.fac * (s / m).sum()
        else:
            if m > 0:
                return self.fac * s / m
            else:
                return jnp.zeros(())

    def prox(self, y, eta):
        y = jnp.maximum(0, y)
        m = y.max(axis=-1, keepdims=True)
        return jnp.maximum(0, jnp.where(y == m, y, y - eta * self.fac / m))
