from logging import getLogger

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np

from ..utils import (
    get_clip,
    get_gpu_env,
)
from .regularizer import L2
from .common import loss_fn
from .dynamics import get_dynamics
from .optimizer import ProxOptimizer
from .penalty import get_penalty
from .prepare import prepare_matrix

logger = getLogger(__name__)


def spatial(data, oldx, y1, y2, dynamics, penalty, env, clip, prepare, optimize, step):
    logger.info("spatial:")
    model = SpatialModel(data, oldx, y1, y2, dynamics, penalty, env)
    out = []
    clip = get_clip(data.shape, *clip)
    for cl in clip:
        index = model.prepare(cl, **prepare)
        optimizer = model.optimizer(**optimize)
        x1, x2 = model.initial_data()
        x1, x2 = optimizer.fit((x1, x2), **step)
        out.append((index, model.unclip(x1, x2)))
    index, x = zip(*out)
    index = np.concatenate(index, axis=0)
    x = np.concatenate(x, axis=0)
    return index, x


def temporal(data, y, peaks, dynamics, penalty, env, prepare, optimize, step):
    logger.info("temporal:")
    model = TemporalModel(data, y, peaks, dynamics, penalty, env)
    model.prepare(**prepare)
    optimizer = model.optimizer(**optimize)
    x1, x2 = model.initial_data()
    x1, x2 = optimizer.fit((x1, x2), **step)
    return np.array(x1), np.array(x2)


class Model:
    def __init__(self, kind, data, dynamics, penalty, env):
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
        nx = jnp.array(nx)
        ny = jnp.array(ny)

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
    def __init__(self, data, oldx, y1, y2, *args, **kwargs):
        self.oldx = oldx
        self.y1 = y1
        self.y2 = y2
        super().__init__("spatial", data, *args, **kwargs)

    def prepare(self, clip, **kwargs):
        logger.info("clip: %s", clip)

        data = self.data.clip(clip)
        trans = False

        oldx = clip(self.oldx)
        n1 = self.y1.shape[0]

        cond = np.any(oldx, axis=(1, 2))
        y1 = jax.device_put(self.y1[cond[:n1]], self.sharding)
        y2 = jax.device_put(self.y2[cond[n1:]], self.sharding)

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
        self._prepare(data, yval, trans, py, bx, by, **kwargs)

        return np.where(cond)[0]

    def unclip(self, x1, x2):
        clip = self._clip
        mask = self._data.mask
        x = np.concatenate([np.array(x1), np.array(x2)], axis=0)
        nk = x.shape[0]
        h, w = self.data.shape
        y0 = 0 if clip.y0 is None else clip.y0
        x0 = 0 if clip.x0 is None else clip.x0
        y1 = h if clip.y1 is None else clip.y1
        x1 = w if clip.x1 is None else clip.x1
        if mask is None:
            tmp = x.reshape(nk, y1 - y0, x1 - x0)
        else:
            tmp = np.zeros((nk, y1 - y0, x1 - x0))
            tmp[mask] = x
        out = np.zeros((nk, h, w))
        print(tmp.shape, out.shape, y0, y1, x0, x1)
        out[:, y0:y1, x0:x1] = tmp
        return out

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
        return self.y[:self.n1]

    @property
    def y2(self):
        return self.y[self.n1:]

    def prepare(self, **kwargs):
        data = self.data
        trans = True

        y = data.apply_mask(self.y, mask_type=True)
        yval = jax.device_put(y, self.sharding)
        y1 = yval[:self.n1]

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
        y2sum = jnp.square(jnp.array(self.y[self.n1:])).sum(axis=(1, 2))
        return self.penalty.lu, L2(lb2 * y2sum[:, jnp.newaxis])
