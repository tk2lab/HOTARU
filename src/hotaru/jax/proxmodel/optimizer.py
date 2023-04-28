from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


class ProxOptimizer:

    def __init__(self, loss_fn, x0, regularizers):
        self.loss_fn = loss_fn
        self.regularizers = regularizers
        self.i = jnp.array(0)
        self.x = [jnp.array(x) for x in x0]
        self.y = [jnp.array(x) for x in x0]

    def set_params(self, lr, scale=20, reset_interval=100):
        self.lr = lr
        self.scale = scale
        self.reset_interval = reset_interval

    def step(self):
        i, x, y = self._update(self.i, self.x, self.y)
        self.i = i.block_until_ready()
        self.x = [xi.block_until_ready() for xi in x]
        self.y = [yi.block_until_ready() for yi in y]
        print(self.val)

    @property
    def val(self):
        return [np.array(x) for x in self.x]

    @property
    def loss(self):
        return self.loss_fn(*self.x)

    @partial(jax.jit, static_argnums=(0,))
    def _update(self, i, x, y):
        scale = self.scale
        lr = self.lr
        j = i % self.reset_interval
        t0 = (scale + j) / scale
        t1 = (scale + 1) / (scale + j + 1)
        grady, aux = jax.grad(self.loss_fn, argnums=range(len(y)), has_aux=True)(*y)
        xy = [
            self._prox_update(ri.prox, xi, yi, gi, lr, t0, t1)
            for ri, xi, yi, gi in zip(self.regularizers, x, y, grady)
        ]
        return j + 1, *zip(*xy)

    def _prox_update(self, prox, oldx, oldy, grady, lr, t0, t1):
        newx = prox(oldy - lr * grady, lr)
        tmpx = (1 - t0) * oldx + t0 * newx
        newy = (1 - t1) * newx + t1 * tmpx
        return newx, newy
