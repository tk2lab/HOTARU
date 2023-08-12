from logging import getLogger

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from .common import mask_fn

logger = getLogger(__name__)


class ProxOptimizer:
    def __init__(self, loss_fn, x0, regularizers, num_devices=1, name=None):
        self.loss_fn = loss_fn
        self.regularizers = regularizers
        self.x = tuple(jnp.array(x) for x in x0)
        self._num_devices = num_devices
        self._name = name

    def set_params(self, lr, scale=20, loss_scale=1):
        self.scale = scale
        self.loss_scale = loss_scale
        self.lr = lr
        self.scale_lr = loss_scale * lr

    @property
    def val(self):
        return tuple(np.array(x) for x in self.x)

    def loss(self):
        loss = self.loss_fn(*self.x)
        penalty = sum(ri(xi) for xi, ri in zip(self.x, self.regularizers))
        return loss + penalty / self.loss_scale

    def fit(self, n_epoch, n_step, tol=None):
        self._grad_loss_fn = jax.grad(self.loss_fn, argnums=range(len(self.x)))
        loss = self.loss()
        log_diff = np.inf
        history = [loss]
        postfix = f"loss={loss:.4f}, diff={log_diff:.2f}"
        logger.info("%s: %s %s %d %s", "pbar", "start", self._name, n_epoch, postfix)
        for i in range(n_epoch):
            self.step(n_step)
            old_loss, loss = loss, self.loss()
            diff = (old_loss - loss) / tol
            log_diff = np.log10(diff) if diff > 0 else np.nan
            history.append(loss)
            postfix = f"loss={loss:.4f}, diff={log_diff:.2f}"
            logger.info("%s: %s %d %s", "pbar", "update", 1, postfix)
            if log_diff < 0.0:
                break
        logger.info("%s: %s", "pbar", "close")
        return history

    def step(self, n_step):
        x = self.x
        x, _ = lax.fori_loop(0, n_step, self._update, (x, x))
        self.x = tuple(xi.block_until_ready() for xi in x)

    def _update(self, i, xy):
        x, y = xy
        scale = self.scale
        t0 = (scale + i) / scale
        t1 = (scale + 1) / (scale + i + 1)
        grady = self._grad_loss_fn(*y)
        xy = [
            self._prox_update(xi, yi, gi, ri.prox, t0, t1)
            for ri, xi, yi, gi in zip(self.regularizers, x, y, grady)
        ]
        return tuple(zip(*xy))

    def _prox_update(self, oldx, oldy, grady, prox, t0, t1):
        def update(oldx, oldy, grady, prox, t0, t1):
            newx = prox(oldy - self.scale_lr * grady, self.lr)
            tmpx = (1 - t0) * oldx + t0 * newx
            newy = (1 - t1) * newx + t1 * tmpx
            return newx, newy

        def mask_update(oldx, oldy, grady, i):
            oldx = mask_fn(oldx, num_devices)[i]
            oldy = mask_fn(oldy, num_devices)[i]
            grady = mask_fn(grady, num_devices)[i]
            return update(oldx, oldy, grady, prox, t0, t1)

        num_devices = self._num_devices
        if num_devices == 1:
            return update(oldx, oldy, grady, prox, t0, t1)
        else:
            dev_id = jnp.arange(num_devices)
            update_pmap = jax.pmap(mask_update, in_axes=(None, None, None, 0))
            newx, newy = update_pmap(oldx, oldy, grady, dev_id)
            return jnp.concatenate(newx, axis=0), jnp.concatenate(newy, axis=0)
