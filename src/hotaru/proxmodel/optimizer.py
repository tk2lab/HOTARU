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
        loss = self.loss_fn(*self.x)
        penalty = sum(ri(xi) for xi, ri in zip(self.x, self.regularizers))
        return loss + penalty

    def fit(self, n_epoch, n_step, tol=None, pbar=None, num_devices=1):
        loss = self.loss()
        diff = np.inf
        history = [loss]
        if pbar is not None:
            pbar.reset(total=n_epoch)
            pbar.set_postfix_str(f"loss={loss:.4f}, diff= nan")
        for i in range(n_epoch):
            self.step(n_step, num_devices)
            old_loss, loss = loss, self.loss()
            diff = np.log10((old_loss - loss) / tol)
            history.append(loss)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"loss={loss:.4f}, diff={diff:.2f}")
            if diff < 0.0:
                break
        return history

    def step(self, n_step, num_devices=1):
        x = self.x
        y = tuple(jnp.array(xi) for xi in x)
        update = partial(self._update, num_devices=num_devices)
        x, _ = lax.fori_loop(0, n_step, update, (x, y))
        self.x = tuple(xi.block_until_ready() for xi in x)

    def _update(self, i, xy, num_devices):
        scale = self.scale
        lr = self.lr
        t0 = (scale + i) / scale
        t1 = (scale + 1) / (scale + i + 1)
        x, y = xy
        grad_loss_fn = jax.grad(self.loss_fn, argnums=range(len(y)))
        grady = grad_loss_fn(*y)
        if num_devices == 1:
            prox_update = self._prox_update
        else:
            prox_update = partial(self._prox_update_pmap, num_devices=num_devices)
        xy = [
            prox_update(xi, yi, gi, ri.prox, lr, t0, t1)
            for ri, xi, yi, gi in zip(self.regularizers, x, y, grady)
        ]
        return tuple(zip(*xy))

    def _prox_update(self, oldx, oldy, grady, prox, lr, t0, t1):
        newx = prox(oldy - lr * grady, lr)
        tmpx = (1 - t0) * oldx + t0 * newx
        newy = (1 - t1) * newx + t1 * tmpx
        return newx, newy

    def _prox_update_pmap(self, oldx, oldy, grady, prox, lr, t0, t1, num_devices):
        def mask_fn(x):
            n, *shape = x.shape
            d = n % num_devices
            xval = jnp.pad(x, [[0, d]] + [[0, 0]] * len(shape))
            return xval.reshape(num_devices, (n + d) // num_devices, *shape)

        def update(oldx, oldy, grady, i):
            oldx = mask_fn(oldx)[i]
            oldy = mask_fn(oldy)[i]
            grady = mask_fn(grady)[i]
            return self._prox_update(oldx, oldy, grady, prox, lr, t0, t1)

        dev_id = jnp.arange(num_devices)
        update_pmap = jax.pmap(update, in_axes=(None, None, None, 0))
        newx, newy = update_pmap(oldx, oldy, grady, dev_id)
        return jnp.concatenate(newx, axis=0), jnp.concatenate(newy, axis=0)
