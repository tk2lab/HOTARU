from keras import Regularizer
from keras import ops

from ..saving import Config


class ProxRegularizer(Regularizer):
    def prox(self, y, lr):
        _ = lr
        return y


class L2Regularizer(ProxRegularizer):
    def __init__(self, fac: float):
        self.fac = fac

    def get_config(self) -> Config:
        return {**super().get_config(), 'fac': self.fac}

    def __call__(self, x):
        return ops.sum(ops.square(self.fac * x))

    def prox(self, y, lr):
        return y / (1 + self.fac * lr)


class L1Regularizer(ProxRegularizer):
    def __init__(self, fac: float):
        self.fac = fac

    def get_config(self) -> Config:
        return {**super().get_config(), 'fac': self.fac}

    def __call__(self, x):
        return self.fac * ops.sum(ops.abs(x))

    def prox(self, y, lr):
        sign = ops.sign(y)
        absy = ops.abs(y)
        return sign * ops.relu(absy - self.fac * lr)


class MaxNormL1Regularizer(ProxRegularizer):
    def __init__(self, fac: float):
        self.fac = fac

    def get_config(self) -> Config:
        return {**super().get_config(), 'fac': self.fac}

    def __call__(self, x):
        absx = ops.abs(x)
        m = ops.max(absx, axis=-1)
        s = ops.sum(absx, axis=-1)
        positive = m > 0
        m = ops.where(positive, m, 1)
        s = ops.where(positive, s, 0)
        return self.fac * ops.sum(s / m)

    def prox(self, y, lr):
        sign = ops.sign(y)
        absy = ops.abs(y)
        m = ops.max(absy, axis=-1, keepdims=True)
        return sign * ops.where(absy == m, absy, ops.relu(absy - self.fac * lr / m))
