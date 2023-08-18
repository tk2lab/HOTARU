from logging import getLogger

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
from .regularizer import (
    L1,
    NonNegativeL1,
)

logger = getLogger(__name__)


class Model:
    def __init__(self, data, trans, stats, dynamics, penalty, env, **kwargs):
        self._dynamics = get_dynamics(dynamics)
        self._penalty = get_penalty(penalty)
        self._env = get_gpu_env(env)

        self._data = data
        self._trans = trans
        self._stats = stats

    def _try_clip(self, clip, segs):
        clipped_segs = clip.clip(segs)
        clipped = np.any(clipped_segs > 0, axis=(1, 2))
        clipped_state = self._stats[clipped]

        n1 = self.n1
        clipped_n1 = np.count_nonzero(clipped[:n1])
        clipped_state1 = clipped_state[:clipped_n1]
        clipped_state2 = clipped_state[clipped_n1:]

        active1 = clip.in_clipping_area(clipped_state1.y, clipped_state1.x)
        active2 = clip.in_clipping_area(clipped_state2.y, clipped_state2.x)
        active_index1 = clipped_state1[active1].index
        active_index2 = clipped_state2[active2].index - n1

        self._clip = clip
        self._active = active1, active2
        self._active_index = active_index1, active_index2

        return clipped, clipped_segs

    def _prepare(self, data, yval, nx1, nx2, py, **kwargs):
        ycov, yout, ydot = prepare_matrix(data, yval, self._trans, self._env, **kwargs)

        penalty = self._penalty
        if self._trans:
            nx, ny = data.nt, data.ns
            bx, by = penalty.bt, penalty.bs
        else:
            nx, ny = data.ns, data.nt
            bx, by = penalty.bs, penalty.bt
        nx, ny, bx, by = (jnp.array(v, jnp.float32) for v in (nx, ny, bx, by))

        active1, active2 = self._active
        n1, n2 = active1.size, active2.size
        x1 = jnp.zeros((n1, nx1))
        x2 = jnp.zeros((n2, nx2))

        self._args = ycov, yout, ydot, nx, ny, bx, by, py
        self._loss_scale = nx * ny + nx + ny

        self._lr_scale = jnp.diagonal(ycov + bx * yout).max()
        self._x = x1, x2

        if not hasattr(self, "_optimizer"):
            self._optimizer = ProxOptimizer(self)


    @property
    def args(self):
        return self._args

    @property
    def loss_scale(self):
        return self._loss_scale

    def loss(self, x1, x2, ycov, yout, ydot, nx, ny, bx, by, py):
        x = jnp.concatenate([x1, x2], axis=0)
        return loss_fn(x, ycov, yout, ydot, nx, ny, bx, by) + py

    @property
    def prox(self):
        return (r.prox for r in self.regularizers)

    def fit(self, max_epoch, steps_par_epoch, lr, *args, **kwargs):
        lr /= self._lr_scale
        x = self._x
        logger.info("%s: %s %s %d", "pbar", "start", "optimize", -1)
        x = self._optimizer.fit(x, max_epoch, steps_par_epoch, lr, *args, **kwargs)
        logger.info("%s: %s", "pbar", "close")
        self._x = x

    def get_x(self):
        active1, active2 = self._active
        active_index1, active_index2 = self._active_index
        x1, x2 = self._x
        active_x1 = np.array(x1[active1])
        active_x2 = np.array(x2[active2])
        return active_index1, active_index2, active_x1, active_x2


class SpatialModel(Model):
    def __init__(self, data, oldx, stats, y1, y2, *args, **kwargs):
        super().__init__(data, False, stats, *args, **kwargs)
        self._oldx = oldx
        self._y1 = y1
        self._y2 = y2

    @property
    def n1(self):
        return self._y1.shape[0]

    @property
    def n2(self):
        return self._y2.shape[0]

    @property
    def regularizers(self):
        return self._penalty.la[0], NonNegativeL1()

    @property
    def prox_args(self):
        return self._penalty.la[1], (self._lb,)

    def try_clip(self, clip):
        return self._try_clip(clip, self._oldx)

    def prepare(self, clip, **kwargs):
        clipped, clipped_segs = self.try_clip(clip)
        clipped_data = self._data.clip(clip.clip)

        n1 = self.n1
        clipped_y1 = self._y1[clipped[:n1]]
        clipped_y2 = self._y2[clipped[n1:]]

        clipped_y1 = jnp.array(clipped_y1)
        clipped_y2 = jnp.array(clipped_y2)

        lu, fac = self._penalty.lu
        py = lu(clipped_y1, *fac)

        clipped_y1 = self._dynamics(clipped_y1)
        clipped_y1 /= clipped_y1.max(axis=1, keepdims=True)
        clipped_y2 /= clipped_y2.max(axis=1, keepdims=True)
        yval = jnp.concatenate([clipped_y1, clipped_y2], axis=0)

        nx1, nx2 = clipped_data.ns, clipped_data.ns
        self._prepare(clipped_data, yval, nx1, nx2, py, **kwargs)

        lb = self._penalty.lb
        self._lb = jnp.abs(lb * clipped_y2).sum(axis=1, keepdims=True)

    def get_x(self):
        index1, index2, x1, x2 = super().get_x()
        index = np.concatenate([index1, index2 + self.n1], axis=0)
        x = np.concatenate([x1, x2], axis=0)

        mask = self._data.mask
        shape = self._data.shape
        x = self._clip.unclip(x, mask, shape)
        return index, x


class TemporalModel(Model):
    def __init__(self, data, y, stats, *args, **kwargs):
        super().__init__(data, True, stats, *args, **kwargs)
        self._y = y

    @property
    def n1(self):
        return np.count_nonzero(self._stats.kind == "cell")

    @property
    def n2(self):
        return np.count_nonzero(self._stats.kind == "background")

    @property
    def prox_args(self):
        return self._penalty.lu[1], (self._lb,)

    def try_clip(self, clip):
        return self._try_clip(clip, self._y)

    def prepare(self, clip, **kwargs):
        clipped, clipped_y = self.try_clip(clip)

        clipped_n1 = np.count_nonzero(clipped[: self.n1])

        data = self._data.clip(clip.clip)
        yval = data.apply_mask(clipped_y[clipped], mask_type=True)

        yval = jnp.array(yval)
        clipped_y1 = yval[:clipped_n1]
        clipped_y2 = yval[clipped_n1:]
        la, fac = self._penalty.la
        py = la(clipped_y1, *fac)

        nt = self._data.nt
        nu = nt + self._dynamics.size - 1
        self._prepare(data, yval, nu, nt, py, **kwargs)

        lb = self._penalty.lb
        self._lb = jnp.abs(lb * clipped_y2).sum(axis=1, keepdims=True)

    def loss(self, x1, x2, ycov, yout, ydot, nx, ny, bx, by, py):
        x1 = self._dynamics(x1)
        return super().loss(x1, x2, ycov, yout, ydot, nx, ny, bx, by, py)

    @property
    def regularizers(self):
        return self._penalty.lu[0], L1()
