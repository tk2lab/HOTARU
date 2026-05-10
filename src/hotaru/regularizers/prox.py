from keras import Regularizer
from keras import Variable
from keras import ops

from ..saving import Config


class ProxRegularizer(Regularizer):
    def prox(self, y, lr):
        _ = lr
        return y


class ProxRegularizerWithFactor(ProxRegularizer):
    def __init__(self, fac: float = 0.0, *, nonneg: bool = False):
        self.fac = Variable(fac, dtype='float32')
        self.nonneg = nonneg

    def get_config(self) -> Config:
        return {**super().get_config(), 'fac': self.fac, 'nonneg': self.nonneg}


class L2Regularizer(ProxRegularizerWithFactor):
    def __call__(self, x):
        return (self.fac / 2) * ops.sum(ops.square(x))

    def prox(self, y, lr):
        return y / (1 + self.fac * lr)


class L1Regularizer(ProxRegularizerWithFactor):
    def __call__(self, x):
        absx = ops.relu(x) if self.nonneg else ops.abs(x)
        return self.fac * absx

    def prox(self, y, lr):
        absy = ops.relu(y) if self.nonneg else ops.abs(y)
        sign = ops.sign(y)
        return sign * ops.relu(absy - self.fac * lr)


class MaxNormL1Regularizer(ProxRegularizerWithFactor):
    def __call__(self, x):
        absx = ops.relu(x) if self.nonneg else ops.abs(x)
        m = ops.max(absx, axis=-1, keepdims=True)
        scalex = absx / ops.where(m > 0, m, 1)
        return self.fac * ops.sum(scalex)

    def prox(self, y, lr):
        absy = ops.relu(y) if self.nonneg else ops.abs(y)
        sign = ops.sign(y)
        m = ops.max(absy, axis=-1, keepdims=True)
        return sign * ops.where(absy == m, absy, ops.relu(absy - self.fac * lr / m))


class SparseShapeRegularizer(ProxRegularizerWithFactor):
    def __call__(self, x):
        absx = ops.relu(x) if self.nonneg else ops.abs(x)
        m = ops.max(absx, axis=-1, keepdims=True)
        scalex = absx / ops.where(m > 0, m, 1)
        return self.fac * ops.sum(scalex - ops.square(scalex) / 2)

    def prox(self, y, lr):
        absy = ops.relu(y) if self.nonneg else ops.abs(y)
        sign = ops.sign(y)
        m = ops.max(absy, axis=-1, keepdims=True)
        tau = self.fac * lr
        q = m * m - tau
        return sign * m * ops.relu(y * m - tau) / ops.where(q > 0, q, 1)
