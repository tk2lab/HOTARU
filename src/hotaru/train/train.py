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
from .regularizer import (
    L1,
    NonNegativeL1,
)

logger = getLogger(__name__)


class Model:
    def __init__(self, data, trans, peaks, dynamics, penalty, env, label="model"):
        self.dynamics = get_dynamics(dynamics)
        self.penalty = get_penalty(penalty)
        self.env = get_gpu_env(env)

        self.data = data
        self.trans = trans
        self.peaks = peaks

        nd = self.env.num_devices
        self.sharding = self.env.sharding((nd, 1))

    def _try_clip(self, clip, seg):
        seg = clip.clip(seg)
        self._clip = clip
        self._clipped = np.any(seg, axis=(1, 2))
        active1, active2, index1, index2 = self._active()
        return index1, index2

    def _prepare(self, clip, clipped, data, yval, py, **kwargs):
        ycov, yout, ydot = prepare_matrix(data, yval, self.trans, self.env, **kwargs)

        penalty = self.penalty
        if self.trans:
            nx, ny = data.nt, data.ns
            bx, by = penalty.bt, penalty.bs
        else:
            nx, ny = data.ns, data.nt
            bx, by = penalty.bs, penalty.bt
        nx, ny, bx, by = (jnp.array(v, jnp.float32) for v in (nx, ny, bx, by))

        self._clip = clip
        self._clipped = clipped
        self._data = data
        self._yval = yval

        self.args = ycov, yout, ydot, nx, ny, bx, by
        self.py = py
        self.lr_scale = jnp.diagonal(ycov + bx * yout).max()
        self.loss_scale = nx * ny + nx + ny

        logger.info(
            "clip: clip=%s data=%s cell=%d background=%d",
            clip,
            data.imgs.shape,
            self._n1,
            self._n2,
        )
        logger.debug("mat scale: %f", np.array(self.lr_scale))

    @property
    def _n1(self):
        return np.count_nonzero(self._clipped[: self.n1])

    @property
    def _n2(self):
        return np.count_nonzero(self._clipped[self.n1 :])

    def _active(self):
        clip = self._clip
        clipped = self._clipped
        clipped_state = self.peaks[clipped]
        clipped_n1 = np.count_nonzero(clipped[: self.n1])
        clipped_state1 = clipped_state[:clipped_n1]
        clipped_state2 = clipped_state[clipped_n1:]
        active1 = clip.in_clipping_area(clipped_state1.y, clipped_state1.x)
        active2 = clip.in_clipping_area(clipped_state2.y, clipped_state2.x)
        active_index1 = clipped_state1[active1].index
        active_index2 = clipped_state2[active2].index - self.n1
        return active1, active2, active_index1, active_index2

    def initial_data(self):
        n1, n2 = self._n1, self._n2
        nx1, nx2 = self._shape
        x1 = jax.device_put(jnp.zeros((n1, nx1)), self.sharding)
        x2 = jax.device_put(jnp.zeros((n2, nx2)), self.sharding)
        return x1, x2

    def finalize(self, x1, x2):
        active1, active2, active_index1, active_index2 = self._active()
        active_x1 = np.array(x1[active1])
        active_x2 = np.array(x2[active2])
        return active_index1, active_index2, active_x1, active_x2

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
        super().__init__(data, False, peaks, *args, **kwargs)
        self.oldx = oldx
        self.y1 = y1
        self.y2 = y2

    @property
    def n1(self):
        return self.y1.shape[0]

    @property
    def n2(self):
        return self.y2.shape[0]

    def try_clip(self, clip):
        return self._try_clip(clip, self.oldx)

    def prepare(self, clip, **kwargs):
        data = self.data.clip(clip.clip)
        oldx = clip.clip(self.oldx)
        clipped = np.any(oldx, axis=(1, 2))

        n1 = self.y1.shape[0]
        y1 = jax.device_put(self.y1[clipped[:n1]], self.sharding)
        y2 = jax.device_put(self.y2[clipped[n1:]], self.sharding)
        py = self.penalty.lu(y1)

        y1 = self.dynamics(y1)
        yval = jnp.concatenate([y1, y2], axis=0)
        yval /= yval.max(axis=1, keepdims=True)

        self._prepare(clip, clipped, data, yval, py, **kwargs)

    @property
    def _shape(self):
        return self._data.ns, self._data.ns

    def regularizer(self):
        lb = self.penalty.lb
        y2 = self._yval[self._n1 :]
        lb = jnp.abs(lb * y2).sum(axis=1, keepdims=True)
        return self.penalty.la, NonNegativeL1(lb)

    def finalize(self, x1, x2):
        index1, index2, x1, x2 = super().finalize(x1, x2)
        index = np.concatenate([index1, index2 + self.n1], axis=0)
        x = np.concatenate([x1, x2], axis=0)

        mask = self._data.mask
        shape = self.data.shape
        x = self._clip.unclip(x, mask, shape)
        return index, x


class TemporalModel(Model):
    def __init__(self, data, y, peaks, *args, **kwargs):
        super().__init__(data, True, peaks, *args, **kwargs)
        self.y = y

    @property
    def n1(self):
        return np.count_nonzero(self.peaks.kind == "cell")

    @property
    def n2(self):
        return np.count_nonzero(self.peaks.kind == "background")

    def try_clip(self, clip):
        return self._try_clip(clip, self.y)

    def prepare(self, clip, **kwargs):
        data = self.data.clip(clip.clip)
        y = clip.clip(self.y)

        clipped = np.any(y, axis=(1, 2))
        yval = data.apply_mask(y[clipped], mask_type=True)
        yval = jax.device_put(yval, self.sharding)

        n1 = np.count_nonzero(clipped[: self.n1])
        py = self.penalty.la(yval[:n1])

        return self._prepare(clip, clipped, data, yval, py, **kwargs)

    @property
    def _shape(self):
        nt = self.data.nt
        nu = nt + self.dynamics.size - 1
        return nu, nt

    def loss_fn(self, x1, x2):
        x1 = self.dynamics(x1)
        return super().loss_fn(x1, x2)

    def regularizer(self):
        lb = self.penalty.lb
        y2 = self._yval[self._n1 :]
        lb = (lb * y2).sum(axis=1, keepdims=True)
        return self.penalty.lu, L1(lb)
