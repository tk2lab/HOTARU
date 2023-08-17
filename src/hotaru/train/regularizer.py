import jax.numpy as jnp


class Regularizer:
    def __call__(self, x):
        return 0

    def prox(self, y, eta):
        return y


class L2:
    def __init__(self, fac):
        self.fac = fac

    def __call__(self, x):
        return jnp.square(self.fac * x).sum()

    def prox(self, y, eta):
        return y / (1 + self.fac * eta)


class L1:
    def __init__(self, fac):
        self.fac = fac

    def __call__(self, x):
        return jnp.abs(self.fac * x).sum()

    def prox(self, y, eta):
        return jnp.sign(y) * jnp.maximum(0, jnp.abs(y) - self.fac * eta)


class NonNegativeL1:
    def __init__(self, fac):
        self.fac = fac

    def __call__(self, x):
        x = jnp.maximum(0, x)
        return (self.fac * x).sum()

    def prox(self, y, eta):
        return jnp.maximum(0, y - self.fac * eta)


class MaxNormNonNegativeL1:
    def __init__(self, fac):
        self.fac = fac

    def __call__(self, x):
        x = jnp.maximum(0, x)
        m = x.max(axis=-1)
        s = (self.fac * x).sum(axis=-1)
        if x.ndim > 1:
            positive = m > 0
            m = jnp.where(positive, m, 1)
            s = jnp.where(positive, s, 0)
            return (s / m).sum()
        else:
            if m > 0:
                return self.fac * s / m
            else:
                return jnp.zeros(())

    def prox(self, y, eta):
        y = jnp.maximum(0, y)
        m = y.max(axis=-1, keepdims=True)
        ym = jnp.where(y == m, y, y - self.fac * eta / m)
        return jnp.maximum(0, ym)
