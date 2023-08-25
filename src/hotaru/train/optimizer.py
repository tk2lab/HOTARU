from logging import getLogger

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

logger = getLogger(__name__)


class ProxOptimizer:
    def __init__(self, model):
        self._model = model
        self._loss_sep_fn = jax.jit(self._loss_sep)
        self._loss_fn = jax.jit(self._loss)
        self._step_fn = jax.jit(self._step, static_argnames=("n_step",))

    def loss(self, *x):
        loss_args = dict(
            args=tuple(jnp.array(v) for v in self._model.args),
            loss_scale=jnp.array(self._model.loss_scale),
            prox_args=tuple(
                tuple(jnp.array(vi) for vi in v) for v in self._model.prox_args
            ),
        )
        x = tuple(jnp.array(xi) for xi in x)
        loss, aux = self._loss_fn(*x, **loss_args)
        return np.array(loss), np.array(aux)

    def step(self, x, lr=1, nesterov=20, n_step=100):
        x = tuple(jnp.array(xi) for xi in x)
        args = tuple(jnp.array(v) for v in self._model.args)
        prox_args = tuple(tuple(jnp.array(vi) for vi in v) for v in self._model.args)
        lr = jnp.array(lr, jnp.float32)
        nesterov = jnp.array(nesterov, jnp.float32)
        return self._step_fn(x, args, prox_args, lr, nesterov, n_step)

    def fit(self, x, max_epoch, steps_par_epoch, lr, nesterov, tol, patience, env):
        common_args = dict(
            args=tuple(jnp.array(v) for v in self._model.args),
            prox_args=tuple(
                tuple(jnp.array(vi) for vi in v) for v in self._model.prox_args
            ),
        )
        loss_args = common_args | dict(loss_scale=jnp.array(self._model.loss_scale))
        step_args = common_args | dict(
            lr=jnp.array(lr, jnp.float32),
            nesterov=jnp.array(nesterov, jnp.float32),
            n_step=steps_par_epoch,
        )

        def loss_fn(*x):
            loss, aux = self._loss_fn(*x, **loss_args)
            return np.array(loss), np.array(aux)

        x = tuple(jnp.array(xi) for xi in x)

        history = []

        loss, aux = loss_fn(*x)
        diff = np.nan
        history.append((loss, aux))

        log_diff = np.inf
        postfix = f"loss={loss:.4f}, diff={log_diff:.2f}"
        logger.info("%s: %s %d %s", "pbar", "update", 0, postfix)

        patience_count = 0
        min_loss = np.inf
        for i in range(max_epoch):
            x = self._step_fn(x, **step_args)

            (loss, aux), old_loss = loss_fn(*x), loss
            diff = (old_loss - loss) / tol
            history.append((loss, aux))

            log_diff = np.log10(diff) if diff > 0 else np.nan
            postfix = f"loss={loss:.4f}, diff={log_diff:.2f}, aux={aux:.4f}"
            logger.info("%s: %s %d %s", "pbar", "update", 1, postfix)

            if loss > min_loss - tol:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count == patience:
                break
            min_loss = min(loss, min_loss)
        return tuple(np.array(xi) for xi in x), history

    def _loss_sep(self, *x, args, loss_scale, prox_args):
        loss, aux = self._model.loss(*x, *args)
        penalty = sum(
            ri(xi, *ai) for xi, ri, ai in zip(x, self._model.regularizers, prox_args)
        )
        return loss / loss_scale, penalty / loss_scale, aux

    def _loss(self, *x, **args):
        loss, penalty, aux = self._loss_sep(*x, **args)
        return loss + penalty, aux

    def _step(self, x, args, prox_args, lr, nesterov, n_step):
        def update(i, xy):
            oldx, oldy = xy
            grady, aux = grad_loss_fn(*oldy, *args)
            t0 = (nesterov + i) / (nesterov)
            t1 = (nesterov + 1) / (nesterov + i + 1)
            newx = tuple(
                p(y - lr * g, lr, *a)
                for p, a, y, g in zip(prox, prox_args, oldy, grady)
            )
            tmpx = tuple((1 - t0) * xo + t0 * xn for xo, xn in zip(oldx, newx))
            newy = tuple((1 - t1) * xn + t1 * xt for xn, xt in zip(newx, tmpx))
            return newx, newy

        grad_loss_fn = jax.grad(self._model.loss, range(len(x)), has_aux=True)
        prox = self._model.prox
        x, _ = lax.fori_loop(0, n_step, update, (x, x))
        return x
