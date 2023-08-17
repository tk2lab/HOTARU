from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import get_gpu_env
from .common import (
    loss_fn,
    prepare_matrix,
)
from .dynamics import get_dynamics
from .optimizer import ProxOptimizer
from .penalty import get_penalty
from .regularizer import L2

logger = getLogger(__name__)


class Model:
    def __init__(self, kind, data, dynamics, penalty, env, label="model"):
        self.kind = kind

        self.dynamics = get_dynamics(dynamics)
        self.penalty = get_penalty(penalty)
        self.env = get_gpu_env(env)

        self.data = data

        nd = self.env.num_devices
        self.sharding = self.env.sharding((nd, 1))

    def _prepare(self, data, y, trans, py, bx, by, **kwargs):
        if trans:
            nx, ny = data.nt, data.ns
        else:
            nx, ny = data.ns, data.nt
        nx = jnp.array(nx, jnp.float32)
        ny = jnp.array(ny, jnp.float32)

        ycov, yout, ydot = prepare_matrix(data, y, trans, self.env, **kwargs)

        self.args = ycov, yout, ydot, nx, ny, bx, by
        self.py = py
        self.lr_scale = jnp.diagonal(ycov + bx * yout).max()
        self.loss_scale = nx * ny + nx + ny

        logger.info("mat scale: %f", np.array(self.lr_scale))

    def optimizer(self, lr, nesterov_scale):
        lr /= self.lr_scale
        loss_scale = self.loss_scale
        pena = self.regularizer()
        optimizer = ProxOptimizer(self.loss_fn, pena, lr, nesterov_scale, loss_scale)
        return optimizer

    def loss_fn(self, x1, x2):
        x = jnp.concatenate([x1, x2], axis=0)
        return loss_fn(x, *self.args) + self.py


class SpatialModel(Model):
    def __init__(self, data, oldx, peaks, y1, y2, *args, **kwargs):
        self.peaks = peaks
        self.oldx = oldx
        self.y1 = y1
        self.y2 = y2
        super().__init__("spatial", data, *args, **kwargs)

    def prepare(self, clip, **kwargs):
        logger.info("clip: %s", clip)

        data = self.data.clip(clip.clip)
        trans = False

        oldx = clip.clip(self.oldx)
        active = np.any(oldx, axis=(1, 2))

        n1 = self.y1.shape[0]
        y1 = jax.device_put(self.y1[active[:n1]], self.sharding)
        y2 = jax.device_put(self.y2[active[n1:]], self.sharding)

        dynamics = self.dynamics
        y1 = dynamics(y1)
        yval = jnp.concatenate([y1, y2], axis=0)
        yval /= yval.max(axis=1, keepdims=True)

        py = self.penalty.lu(y1)
        bx = self.penalty.bs
        by = self.penalty.bt

        self._data = data
        self._clip = clip
        self._y1 = y1
        self._y2 = y2
        self._active = active
        self._prepare(data, yval, trans, py, bx, by, **kwargs)

    def finalize(self, x1, x2):
        shape = self.data.shape
        clip = self._clip
        mask = self._data.mask

        stats = self.peaks[self._active]
        select = clip.in_clipping_area(stats.y, stats.x)
        select_stats = stats[select]

        x = np.concatenate([np.array(x1), np.array(x2)], axis=0)
        select_x = x[select]
        select_x = clip.unclip(select_x, mask, shape)
        return select_stats.index, select_x

    def initial_data(self):
        n1 = self._y1.shape[0]
        n2 = self._y2.shape[0]
        ns = self._data.ns
        return jnp.zeros((n1, ns)), jnp.zeros((n2, ns))

    def regularizer(self):
        lb2 = jnp.square(self.penalty.lb)
        y2sum = jnp.square(jnp.array(self._y2)).sum(axis=1)
        return self.penalty.la, L2(lb2 * y2sum[:, jnp.newaxis])


class TemporalModel(Model):
    def __init__(self, data, y, peaks, *args, **kwargs):
        self.n1 = np.count_nonzero(peaks.kind == "cell")
        self.y = y
        self.peaks = peaks
        super().__init__("temporal", data, *args, **kwargs)

    @property
    def y1(self):
        return self.y[: self.n1]

    @property
    def y2(self):
        return self.y[self.n1 :]

    def prepare(self, **kwargs):
        data = self.data
        trans = True

        y = data.apply_mask(self.y, mask_type=True)
        yval = jax.device_put(y, self.sharding)
        y1 = yval[: self.n1]

        py = self.penalty.la(y1)
        bx = self.penalty.bt
        by = self.penalty.bs

        self._prepare(data, yval, trans, py, bx, by, **kwargs)

    def initial_data(self):
        nk = self.y.shape[0]
        n1 = np.count_nonzero(self.peaks.kind == "cell")
        n2 = nk - n1
        nt = self.data.nt
        nu = nt + self.dynamics.size - 1
        x1 = jax.device_put(jnp.zeros((n1, nu)), self.sharding)
        x2 = jax.device_put(jnp.zeros((n2, nt)), self.sharding)
        return x1, x2

    def loss_fn(self, x1, x2):
        x1 = self.dynamics(x1)
        return super().loss_fn(x1, x2)

    def regularizer(self):
        lb2 = jnp.square(self.penalty.lb)
        y2sum = jnp.square(jnp.array(self.y[self.n1 :])).sum(axis=(1, 2))
        return self.penalty.lu, L2(lb2 * y2sum[:, jnp.newaxis])
