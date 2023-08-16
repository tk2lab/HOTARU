from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np

logger = getLogger(__name__)


class ProxOptimizer:
    def __init__(self, loss_fn, regularizers, lr, nesterov_scale=20, loss_scale=1):
        logger.info("optimizer: %g", lr)
        self.loss_fn = loss_fn
        self.prox = [ri.prox for ri in regularizers]
        self.regularizers = regularizers
        self.nesterov_scale = jnp.array(nesterov_scale, jnp.float32)
        self.lr = jnp.array(lr, jnp.float32)
        self.loss_scale = loss_scale
        self._history = []

    def fit(self, x, max_epoch, steps_par_epoch, tol, patience, name="fit"):
        loss_fn = jax.jit(self._loss)
        step_fn = jax.jit(self._step, static_argnames=("n_step",))

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
            old_loss, loss = loss, np.array(loss_fn(*x))
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

    def step(self, x, n_step):
        x = tuple(jnp.array(xi) for xi in x)
        x = self._step(x, n_step)
        x = tuple(np.array(xi) for xi in x)
        return x

    def loss(self, *x):
        return np.array(jax.jit(self._loss)(*(jnp.array(xi) for xi in x)))

    def _step(self, x, n_step):
        prox = self.prox
        nesterov_scale = self.nesterov_scale
        lr = self.lr
        grad_loss_fn = jax.grad(self.loss_fn, range(len(x)))
        oldx, oldy = x, x
        for i in range(n_step):
            grady = grad_loss_fn(*oldy)
            t0 = (nesterov_scale + i) / (nesterov_scale)
            t1 = (nesterov_scale + 1) / (nesterov_scale + i + 1)
            newx = tuple(p(y - lr * g, lr) for p, y, g in zip(prox, oldy, grady))
            tmpx = tuple((1 - t0) * xo + t0 * xn for xo, xn in zip(oldx, newx))
            newy = tuple((1 - t1) * xn + t1 * xt for xn, xt in zip(newx, tmpx))
            oldx, oldy = newx, newy
        return newx

    def _loss(self, *x):
        loss = self.loss_fn(*x)
        penalty = sum(ri(xi) for xi, ri in zip(x, self.regularizers))
        return (loss + penalty) / self.loss_scale
