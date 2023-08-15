from logging import getLogger

import jax.numpy as jnp
import numpy as np

from ..utils import (
    get_clip,
    get_gpu_env,
)
from .common import loss_fn
from .dynamics import get_dynamics
from .optimizer import ProxOptimizer
from .penalty import get_penalty
from .prepare import prepare_matrix

logger = getLogger(__name__)


def spatial(data, oldx, y1, y2, dynamics, penalty, env, clip, prepare, optimize, step):
    logger.info("spatial:")
    model = SpatialModel(data, oldx, y1, y2, dynamics, penalty, env)
    out = {}
    for cl in clip:
        model.prepare(cl, **prepare)
        optimizer = model.optimizer(**optimize)
        x1, x2 = model.initial_data()
        x1, x2 = optimizer.fit((x1, x2), **step)
        out[cl] = np.concatenate([np.array(x1), np.array(x2)], axis=0)
    return out


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

    def _prepare(self, data, y, trans, bx, by, **kwargs):
        ycov, yout, ycor = prepare_matrix(data, y, trans, self.env, **kwargs)

        cx = 1 - jnp.square(bx)
        cy = 1 - jnp.square(by)

        a = ycor
        b = ycov - cx * yout
        c = yout - cy * ycov

        nt = data.nt
        ns = data.ns
        ntf = jnp.array(nt, jnp.float32)
        nsf = jnp.array(ns, jnp.float32)
        nn = ntf * nsf
        nm = nn + ntf + nsf
        self.args = nn, nm, a, b, c
        self.lr_scale = b.diagonal().max()
        self.loss_scale = nm

    def optimizer(self, lr, nesterov_scale):
        lr /= self.lr_scale
        loss_scale = self.loss_scale
        pena = self.regularizer()
        optimizer = ProxOptimizer(self.loss_fn, pena, lr, nesterov_scale, loss_scale)
        return optimizer

    def loss_fn(self, x1, x2):
        x = jnp.concatenate([x1, x2], axis=0)
        return loss_fn(x, *self.args) + self.py / self.loss_scale


class SpatialModel(Model):
    def __init__(self, data, oldx, y1, y2, *args, **kwargs):
        self.oldx = oldx
        self.y1 = y1
        self.y2 = y2
        super().__init__("spatial", data, *args, **kwargs)

    def prepare(self, clip, **kwargs):
        print(clip)
        clip = get_clip(clip, self.data.shape)
        print(clip)
        data = self.data.clip(clip)
        oldx = clip(self.oldx)
        print(oldx.shape)

        penalty = self.penalty
        dynamics = self.dynamics
        y1 = dynamics(self.y1)
        y2 = self.y2
        yval = jnp.concatenate([y1, y2], axis=0)
        yval /= yval.max(axis=1, keepdims=True)
        trans = False
        bx = penalty.bs
        by = penalty.bt
        self._prepare(data, yval, trans, bx, by, **kwargs)
        self.py = penalty.lu(y1) + penalty.lt(y2)

    def initial_data(self):
        n1 = self.y1.shape[0]
        n2 = self.y2.shape[0]
        ns = self.data.ns
        return jnp.zeros((n1, ns)), jnp.zeros((n2, ns))

    def regularizer(self):
        return self.penalty.la, self.penalty.ls


class TemporalModel(Model):
    def __init__(self, data, y, peaks, *args, **kwargs):
        self.y = y
        self.peaks = peaks
        super().__init__("temporal", data, *args, **kwargs)

    def prepare(self, **kwargs):
        penalty = self.penalty
        data = self.data
        yval = data.apply_mask(self.y, mask_type=True)
        trans = True
        bx = penalty.bt
        by = penalty.bs
        self._prepare(self.data, yval, trans, bx, by, **kwargs)

        nk = np.count_nonzero(self.peaks.kind == "cell")
        self.py = penalty.la(yval[:nk]) + penalty.ls(yval[nk:])

    def initial_data(self):
        nk = self.y.shape[0]
        n1 = np.count_nonzero(self.peaks.kind == "cell")
        n2 = nk - n1
        nt = self.data.nt
        nu = nt + self.dynamics.size - 1
        return jnp.zeros((n1, nu)), jnp.zeros((n2, nt))

    def loss_fn(self, x1, x2):
        x1 = self.dynamics(x1)
        return super().loss_fn(x1, x2)

    def regularizer(self):
        return self.penalty.la, self.penalty.ls
