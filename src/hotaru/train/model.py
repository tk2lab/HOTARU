from logging import getLogger

import jax.numpy as jnp
import numpy as np

from .common import (
    loss_fn,
    prepare_matrix,
)
from ..spike import get_dynamics
from .optimizer import ProxOptimizer
from .penalty import get_penalty
from .regularizer import (
    L1,
    NonNegativeL1,
)

logger = getLogger(__name__)


class Model:
    def __init__(self, data, trans, stats, dynamics, penalty, **kwargs):
        self._dynamics = get_dynamics(dynamics)
        self._penalty = get_penalty(penalty)

        self._data = data
        self._trans = trans
        self._stats = stats

    def _try_clip(self, clip, segs):
        cdf = self._stats.query("kind=='cell'")
        bdf = self._stats.query("kind=='background'")
        cid = cdf.segid.to_numpy().astype(np.int32)
        bid = bdf.segid.to_numpy().astype(np.int32)
        fp = segs[cid]
        bg = segs[bid]

        clipped_fp = clip.clip(fp)
        clipped1 = np.any(clipped_fp > 0, axis=(1, 2))
        clipped_cdf = cdf[clipped1]
        clipped_fp = clipped_fp[clipped1]

        clipped_bg = clip.clip(bg)
        clipped2 = np.any(clipped_bg > 0, axis=(1, 2))
        clipped_bdf = bdf[clipped2]
        clipped_bg = clipped_bg[clipped2]

        active1 = clip.in_clipping_area(clipped_cdf.y, clipped_cdf.x)
        active2 = clip.in_clipping_area(clipped_bdf.y, clipped_bdf.x)

        self._clip = clip
        self._active = active1, active2

        index1 = np.where(clipped1)[0][active1]
        index2 = np.where(clipped2)[0][active2]
        self._active_index = index1, index2

        return clipped1, clipped2, clipped_fp, clipped_bg

    def _prepare(self, data, yval, py, **kwargs):
        ycov, yout, ydot = prepare_matrix(data, yval, self._trans, **kwargs)

        penalty = self._penalty
        if self._trans:
            nx, ny = data.nt, data.ns
            bx, by = penalty.bt, penalty.bs
            logger.info("lu: %f", self._penalty.lu[1][0])
        else:
            nx, ny = data.ns, data.nt
            bx, by = penalty.bs, penalty.bt
            logger.info("la: %f", self._penalty.la[1][0])
        nx, ny, bx, by = (jnp.array(v, jnp.float32) for v in (nx, ny, bx, by))

        self._args = ycov, yout, ydot, nx, ny, bx, by, py
        self._loss_scale = nx * ny + nx + ny
        self._penalty_scale = ny

        est = np.linalg.eigh(ycov)[0].max()
        if not np.isfinite(est):
            est = 1.0
        self._lr_scale = est / nx

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
        loss, sigma = loss_fn(x, ycov, yout, ydot, nx, ny, bx, by)
        return loss + py, sigma

    @property
    def prox(self):
        return (r.prox for r in self.regularizers)

    def fit(self, max_epoch, steps_par_epoch, lr, *args, **kwargs):
        logger.info("fit: lr=%g scale=%f", lr, self._lr_scale)
        lr /= self._lr_scale
        x = self._x
        logger.info("%s: %s %s %d", "pbar", "start", "optimize", -1)
        x, history = self._optimizer.fit(
            x, max_epoch, steps_par_epoch, lr, *args, **kwargs
        )
        logger.info("%s: %s", "pbar", "close")
        self._x = x
        return history

    def get_x(self):
        index1, index2 = self._active_index
        x1, x2 = self._x
        active1, active2 = self._active
        active_x1 = np.array(x1[active1])
        active_x2 = np.array(x2[active2])
        return index1, index2, active_x1, active_x2


class SpatialModel(Model):
    def __init__(self, data, stats, oldx, y1, y2, *args, **kwargs):
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
        return (self._penalty_scale * self._penalty.la[1][0],), (self._lb,)

    def try_clip(self, clip):
        return self._try_clip(clip, self._oldx)

    def prepare(self, clip, **kwargs):
        clipped1, clipped2, clipped_img1, clipped_img2 = self.try_clip(clip)
        data = self._data.clip(clip.clip)

        clipped_y1 = self._y1[clipped1]
        clipped_y2 = self._y2[clipped2]

        clipped_y1 = jnp.array(clipped_y1)
        clipped_y2 = jnp.array(clipped_y2)

        lu, fac = self._penalty.lu
        py = data.nt * lu(clipped_y1, *fac)

        lb = self._penalty.lb
        self._lb = jnp.abs(lb * clipped_y2).sum(axis=1, keepdims=True)

        clipped_y1 = self._dynamics(clipped_y1)
        clipped_y1 /= clipped_y1.max(axis=1, keepdims=True)
        clipped_y2 /= clipped_y2.max(axis=1, keepdims=True)
        yval = jnp.concatenate([clipped_y1, clipped_y2], axis=0)
        self._prepare(data, yval, py, **kwargs)

        clipped_x1 = data.apply_mask(clipped_img1, mask_type=True)
        clipped_x2 = data.apply_mask(clipped_img2, mask_type=True)
        x1 = jnp.array(clipped_x1)
        x2 = jnp.array(clipped_x2)
        #x1 = jnp.zeros_like(clipped_x1)
        #x2 = jnp.zeros_like(clipped_x2)
        self._x = x1, x2

    def get_x(self):
        index1, index2, x1, x2 = super().get_x()

        index = np.concatenate([index1, index2 + self.n1], axis=0)
        x = np.concatenate([x1, x2], axis=0)

        mask = self._data.mask
        shape = self._data.shape
        x = self._clip.unclip(x, mask, shape)
        return index, x


class TemporalModel(Model):
    def __init__(self, data, stats, y, *args, **kwargs):
        super().__init__(data, True, stats, *args, **kwargs)
        self._y = y

    @property
    def n1(self):
        return np.count_nonzero(self._stats.kind == "cell")

    @property
    def n2(self):
        return np.count_nonzero(self._stats.kind == "background")

    @property
    def regularizers(self):
        return self._penalty.lu[0], L1()

    @property
    def prox_args(self):
        return (self._penalty_scale * self._penalty.lu[1][0],), (self._lb,)

    def try_clip(self, clip):
        return self._try_clip(clip, self._y)

    def prepare(self, clip, **kwargs):
        _, _, clipped_y1, clipped_y2 = self.try_clip(clip)
        data = self._data.clip(clip.clip)

        clipped_y1 = data.apply_mask(clipped_y1, mask_type=True)
        clipped_y2 = data.apply_mask(clipped_y2, mask_type=True)

        clipped_y1 = jnp.array(clipped_y1)
        clipped_y2 = jnp.array(clipped_y2)

        la, fac = self._penalty.la
        py = la(clipped_y1, *fac)

        lb = self._penalty.lb
        self._lb = jnp.abs(lb * clipped_y2).sum(axis=1, keepdims=True)

        yval = jnp.concatenate([clipped_y1, clipped_y2], axis=0)
        self._prepare(data, yval, py, **kwargs)

        active1, active2 = self._active
        n1, n2 = active1.size, active2.size
        nt = self._data.nt
        nu = nt + self._dynamics.size - 1
        x1 = jnp.zeros((n1, nu))
        x2 = jnp.zeros((n2, nt))
        self._x = x1, x2

    def loss(self, x1, x2, ycov, yout, ydot, nx, ny, bx, by, py):
        x1 = self._dynamics(x1)
        return super().loss(x1, x2, ycov, yout, ydot, nx, ny, bx, by, py)
