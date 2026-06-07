from functools import partial
from functools import wraps
from math import prod

import keras
from jax import Array
from keras import ops
from keras.backend import backend
from keras.random import SeedGenerator

from .typing import Tensor


class Generator:
    def __init__(self, seed: SeedGenerator | Tensor | Array | int | None = None):
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
    def state(self) -> Tensor:
        return self.seed_gen.state.value

    @state.setter
    def state(self, val) -> None:
        self.seed_gen.state.assign(val)

    def gen_seed(self) -> int:
        return int(keras.random.randint((), 1, 2 * 30, seed=self.seed_gen))

    def categorical(self, logits, shape=(), dtype='int32') -> Tensor:
        logits = ops.convert_to_tensor(logits)
        *batch_shape, num_category = logits.shape
        flat_logits = ops.reshape(logits, (-1, num_category))
        flat_out = keras.random.categorical(flat_logits, prod(shape), dtype, seed=self.seed_gen)
        return ops.reshape(flat_out, (*batch_shape, *shape))

    def normal(self, m=0, s=1, shape=(), dtype='float32') -> Tensor:
        return keras.random.normal(shape, m, s, dtype, seed=self.seed_gen)

    def exponential(self, m=1, b=0, shape=(), dtype='float32'):
        return m * -ops.log(self.uniform(0, ops.exp(-b / m), shape=shape, dtype=dtype))

    def gamma(self, m=1, s=1, shape=(), dtype='float32') -> Tensor:
        normalized_gamma_samples = s * keras.random.gamma(shape, 1 / s, dtype, seed=self.seed_gen)
        return m * normalized_gamma_samples

    def invgamma(self, m=1, s=1, shape=(), dtype='float32') -> Tensor:
        normalized_gamma_samples = s * self.gamma(1 / s, shape=shape, dtype=dtype)
        return m / ops.maximum(normalized_gamma_samples, 1e-7)

    def invgauss(self, m, s, shape=(), dtype='float32') -> Tensor:
        u = self.uniform(shape=shape, dtype=dtype)
        v = s * ops.square(self.normal(shape=shape, dtype=dtype)) / 2
        x = 1 + v - ops.sqrt(2 * v + ops.square(v))
        condition = u <= 1 / (1 + x)
        return m * ops.where(condition, x, 1 / ops.maximum(x, 1e-7))

    def __getattr__(self, name):
        def wrap(*args, shape=(), **kwargs) -> Tensor:
            return getattr(keras.random, name)(shape, *args, seed=self.seed_gen, **kwargs)

        return wrap


def with_rng(*static_argnames):
    def dec(fn):
        @wraps(fn)
        def wrap_func(*args, rng: Generator, **kwargs) -> Tensor:
            func = fn
            if backend() == 'jax':
                import jax

                func = partial(jax.jit, static_argnames=static_argnames)(fn)
            val, state = func(*args, **kwargs, random_state=rng.state)
            rng.state = state
            return val

        return wrap_func

    return dec
