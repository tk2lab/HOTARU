from math import prod

import keras
from jax import Array
from keras import KerasTensor
from keras import ops
from keras.random import SeedGenerator


class Generator:
    def __init__(self, seed: SeedGenerator | KerasTensor | Array | int | None = None):
        match seed:
            case SeedGenerator() as seed_gen:
                self.seed_gen = seed_gen
            case int(seed):
                self.seed_gen = SeedGenerator(seed)
            case None:
                self.seed_gen = SeedGenerator()
            case state:
                self.seed_gen = SeedGenerator()
                self.seed_gen.state.assign(state)

    @property
    def state(self):
        return self.seed_gen.state.value

    @state.setter
    def state(self, val):
        self.seed_gen.state.assign(val)

    def gen_seed(self) -> int:
        return int(keras.random.randint((), 1, 2*30, seed=self.seed_gen))

    def normal(self, m=0, s=1, shape=(), dtype='float32'):
        return keras.random.normal(shape, m, s, dtype, seed=self.seed_gen)

    def categorical(self, logits, shape=(), dtype='int32'):
        out_shape = logits.shape[:-1] + shape
        logits = ops.reshape(logits, (-1, logits.shape[-1]))
        size = prod(shape)
        out = keras.random.categorical(logits, size, dtype, seed=self.seed_gen)
        return ops.reshape(out, out_shape)

    def invgamma(self, m=1, s=1, shape=(), dtype='float32'):
        normalized_gamma_samples = s * self.gamma(1 / s, shape=shape, dtype=dtype)
        return m / ops.where(normalized_gamma_samples > 0, normalized_gamma_samples, 1e-7)

    def invgauss(self, m, s, shape=(), dtype='float32'):
        u = self.uniform(shape=shape, dtype=dtype)
        v = self.normal(shape=shape, dtype=dtype)
        v2 = ops.square(v) / 2
        s2 = ops.square(s)
        x1 = 1 + s2 * v2
        x1 -= s2 * ops.sqrt(v2 / s2 + ops.square(v2))
        condition = u <= 1 / (1 + x1)
        return m * ops.where(condition, x1, 1 / ops.where(x1 > 0, x1, 1e-7))

    def __getattr__(self, name):
        def wrap(*args, shape=(), **kwargs):
            return getattr(keras.random, name)(shape, *args, seed=self.seed_gen, **kwargs)

        return wrap
