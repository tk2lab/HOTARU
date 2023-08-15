from logging import getLogger
from functools import partial

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

logger = getLogger(__name__)


class ProxOptimizer:
    def __init__(self, loss_fn, regularizers, lr, nesterov_scale=20, loss_scale=1):
        self.loss_fn = loss_fn
        self.prox = [ri.prox for ri in regularizers]
        self.regularizers = regularizers
        self.nesterov_scale = jnp.array(nesterov_scale, jnp.float32)
        self.lr = jnp.array(lr, jnp.float32)
        self.loss_scale = jnp.array(loss_scale, jnp.float32)
        self._history = []

    def fit(self, x0, n_epoch, n_step, tol, name="fit"):
        x = tuple(jnp.array(x) for x in x0)
        loss = np.array(self._loss(*x))
        log_diff = np.inf
        self._history.append(loss)
        postfix = f"loss={loss:.4f}, diff={log_diff:.2f}"
        logger.info("%s: %s %s %d %s", "pbar", "start", name, n_epoch, postfix)
        for i in range(n_epoch):
            x = self.step(x, n_step)
            old_loss, loss = loss, np.array(self._loss(*x))
            self._history.append(loss)
            diff = (old_loss - loss) / tol
            log_diff = np.log10(diff) if diff > 0 else np.nan
            postfix = f"loss={loss:.4f}, diff={log_diff:.2f}"
            logger.info("%s: %s %d %s", "pbar", "update", 1, postfix)
            if log_diff < 0.0:
                break
        logger.info("%s: %s", "pbar", "close")
        return tuple(np.array(xi) for xi in x)

    def step(self, x, n_step):
        x = tuple(jnp.array(xi) for xi in x)
        x = self._step(x, n_step)
        x = tuple(np.array(xi) for xi in x)
        return x

    def loss(self, *x):
        return np.array(self._loss(*(jnp.array(xi) for xi in x)))

    @partial(jax.jit, static_argnums=(0, 2))
    def _step(self, x, n_step):
        prox = self.prox
        nesterov_scale = self.nesterov_scale
        lr = self.lr
        scale_lr = self.loss_scale * lr
        grad_loss_fn = jax.grad(self.loss_fn, range(len(x)))
        oldx, oldy = x, x
        for i in range(n_step):
            grady = grad_loss_fn(*oldy)
            t0 = (nesterov_scale + i) / (nesterov_scale)
            t1 = (nesterov_scale + 1) / (nesterov_scale + i + 1)
            newx = tuple(p(y - scale_lr * g, lr) for p, y, g in zip(prox, oldy, grady))
            tmpx = tuple((1 - t0) * xo + t0 * xn for xo, xn in zip(oldx, newx))
            newy = tuple((1 - t1) * xn + t1 * nt for xn, nt in zip(newx, tmpx))
            oldx, oldy = newx, newy
        return newx

    @partial(jax.jit, static_argnums=(0,))
    def _loss(self, *x):
        loss = self.loss_fn(*x)
        penalty = sum(ri(xi) for xi, ri in zip(x, self.regularizers))
        return loss + penalty / self.loss_scale
