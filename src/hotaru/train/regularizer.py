import jax.numpy as jnp


class Regularizer:
    def __call__(self, x):
        return 0

    def prox(self, y, eta):
        return y


class L2:
    def __call__(self, x, fac):
        return jnp.square(fac * x).sum()

    def prox(self, y, eta, fac):
        return y / (1 + fac * eta)


class NonNegativeL2:
    def __call__(self, x, fac):
        return jnp.square(fac * x).sum()

    def prox(self, y, eta, fac):
        return jnp.maximum(0, y) / (1 + fac * eta)


class L1:
    def __call__(self, x, fac):
        return jnp.abs(fac * x).sum()

    def prox(self, y, eta, fac):
        return jnp.sign(y) * jnp.maximum(0, jnp.abs(y) - fac * eta)


class NonNegativeL1:
    def __call__(self, x, fac):
        x = jnp.maximum(0, x)
        return (fac * x).sum()

    def prox(self, y, eta, fac):
        return jnp.maximum(0, y - fac * eta)


class MaxNormNonNegativeL1:
    def __call__(self, x, fac):
        x = jnp.maximum(0, x)
        m = x.max(axis=-1)
        s = (fac * x).sum(axis=-1)
        if x.ndim > 1:
            positive = m > 0
            m = jnp.where(positive, m, 1)
            s = jnp.where(positive, s, 0)
            return (s / m).sum()
        else:
            if m > 0:
                return s / m
            else:
                return jnp.zeros(())

    def prox(self, y, eta, fac):
        y = jnp.maximum(0, y)
        m = y.max(axis=-1, keepdims=True)
        ym = jnp.where(y == m, y, y - fac * eta / m)
        return jnp.maximum(0, ym)
