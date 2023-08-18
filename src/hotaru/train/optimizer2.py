from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np

from .regularizer import L1

logger = getLogger(__name__)


class ProxOptimizer:
    def __init__(self, model, lr=1, nesterov_scale=20):
        self._model = model
        self._lr = jnp.array(lr, jnp.float32)
        self._nesterov_scale = jnp.array(nesterov_scale, jnp.float32)
        self._loss = jax.jit(self._loss, static_argnums=("args", "loss_scale"))
        self._step = jax.jit(self._step, static_argnames=("n_step",))

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, val):
        self._lr = jnp.array(val, jnp.float32)

    @property
    def nesterov(self):
        return self._lr

    @nesterov.setter
    def nesterov(self, val):
        self._nesterov = jnp.array(val, jnp.float32)

    def _loss(self, *x, args, loss_scale):
        model = self.model
        loss = model.loss(*x, *args)
        penalty = sum(ri(xi) for xi, ri in zip(x, model.regularizers))
        return (loss + penalty) / loss_scale

    def loss(self, *x):
        return self._loss(*x, *self._model.args)

    def _step(self, x, args, lr, nesterov, n_step):
        def update(i, args):
            oldx, oldy, args = args
            grady = grad_loss_fn(*oldy, *args)
            t0 = (nesterov + i) / (nesterov)
            t1 = (nesterov + 1) / (nesterov + i + 1)
            newx = tuple(p(y - lr * g, lr) for p, y, g in zip(prox, oldy, grady))
            tmpx = tuple((1 - t0) * xo + t0 * xn for xo, xn in zip(oldx, newx))
            newy = tuple((1 - t1) * xn + t1 * xt for xn, xt in zip(newx, tmpx))
            return newx, newy

        model = self.model
        prox = model.prox
        grad_loss_fn = jax.grad(self._loss, range(len(x)))
        x, *_ = lax.fori_loop(0, n_step, update, (x, x, args))
        return x

    def step(self, x, n_step=100):
        model = self.model
        return self._step(x, model.args, self.lr, self.nesterov, n_step)

    def fit(self, x, max_epoch, steps_par_epoch, tol, patience, name="fit"):
        model = self.model
        args = model.args
        loss_fn = self.loss
        step_fn = self.step
        x = tuple(jnp.array(xi) for xi in x)
        loss = np.array(loss_fn(*x))
        log_diff = np.inf
        self._history.append(loss)
        postfix = f"loss={loss:.4f}, diff={log_diff:.2f}"
        logger.info("%s: %s %s %d %s", "pbar", "start", name, -1, postfix)
        min_loss = np.inf
        patience_count = 0
        for i in range(max_epoch):
            x = step_fn(x, steps_par_epoch)
            old_loss, loss = loss, np.array(loss_fn(*x, *args))
            self._history.append(loss)
            diff = (old_loss - loss) / tol
            log_diff = np.log10(diff) if diff > 0 else np.nan
            postfix = f"loss={loss:.4f}, diff={log_diff:.2f}"
            logger.info("%s: %s %d %s", "pbar", "update", 1, postfix)
            if loss > min_loss - tol:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count == patience:
                break
            min_loss = min(loss, min_loss)
        logger.info("%s: %s", "pbar", "close")
        return tuple(np.array(xi) for xi in x)


class Model:

    def __init__(self):
        self._a = jnp.ones((5, 5))
        self._b = jnp.ones((5,))
        self._loss_scale = jnp.ones(())
        self.regularizer = L1(10)
        self.prox = self.regularizer.prox

    def loss(self, x, a, b):
        return (a * np.outer(x, x)).sum() + np.inner(b, x)

    @property
    def args(self):
        return self._a, self._b, self._loss_scale


model = Model()
x = jnp.ones((5))
optimizer = ProxOptimizer(model)
x = optimizer.fit(x, 5, 5, 1e-5, 100)

model._a = 0.1 * jnp.ones((5, 5))
x = optimizer.fit(x, 5, 5, 1e-5, 100)
