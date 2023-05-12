from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np


class ProxOptimizer:

    def __init__(self, loss_fn, x0, regularizers):
        self.loss_fn = loss_fn
        self.regularizers = regularizers
        self.x = tuple(jnp.array(x) for x in x0)

    def set_params(self, lr, scale=20):
        self.lr = lr
        self.scale = scale

    @property
    def val(self):
        return tuple(np.array(x) for x in self.x)

    def loss(self):
        loss, aux = self.loss_fn(*self.x)
        penalty = sum(ri(xi) for xi, ri in zip(self.x, self.regularizers))
        return loss + penalty, aux

    def fit(self, n_epoch, n_step, tol=None, pbar=None):
        loss, aux = self.loss()
        diff = np.inf
        history = [(loss, aux)]
        if pbar is not None:
            pbar = pbar(total=n_epoch)
            pbar.set_description("optimize")
            pbar.set_postfix(dict(loss=f"{loss:.4f}", diff=" nan"))
        for i in range(n_epoch):
            self.step(n_step)
            old_loss, loss, aux = loss, *self.loss()
            diff = np.log10((old_loss - loss) / tol)
            history.append((loss, aux))
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(dict(loss=f"{loss:.4f}", diff=f"{diff:.2f}"))
            if diff < 0.0:
                break
        return history

    def step(self, n_step):
        x = self.x
        y = tuple(jnp.array(xi) for xi in x)
        x, _ = lax.fori_loop(0, n_step, self._update, (x, y))
        self.x = tuple(xi.block_until_ready() for xi in x)

    def _update(self, i, xy):
        scale = self.scale
        lr = self.lr
        t0 = (scale + i) / scale
        t1 = (scale + 1) / (scale + i + 1)
        x, y = xy
        grad_loss_fn = jax.grad(self.loss_fn, argnums=range(len(y)), has_aux=True)
        grady, aux = grad_loss_fn(*y)
        xy = [
            self._prox_update(ri.prox, xi, yi, gi, lr, t0, t1)
            for ri, xi, yi, gi in zip(self.regularizers, x, y, grady)
        ]
        return tuple(zip(*xy))

    def _prox_update(self, prox, oldx, oldy, grady, lr, t0, t1):
        newx = prox(oldy - lr * grady, lr)
        tmpx = (1 - t0) * oldx + t0 * newx
        newy = (1 - t1) * newx + t1 * tmpx
        return newx, newy
