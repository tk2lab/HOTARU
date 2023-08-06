from collections import namedtuple

import jax.numpy as jnp
import numpy as np

from ..utils import get_progress
from .common import (
    calc_cov_out,
    matmul_batch,
)
from .dynamics import get_dynamics
from .penalty import get_penalty

Prepare = namedtuple("Prepare", "nt nx nk pena a b c")


def prepare_temporal(footprint, data, dynamics, penalty, batch, pbar=None):
    pbar = get_progress(pbar)
    pbar.session("temporal prepare")
    return gen_factor("temporal", footprint, data, dynamics, penalty, batch, pbar)


def prepare_spatial(spike, data, dynamics, penalty, batch, pbar=None):
    pbar = get_progress(pbar)
    pbar.session("spatioal prepare")
    return gen_factor("spatial", spike, data, dynamics, penalty, batch, pbar)


def gen_factor(kind, yval, data, dynamics, penalty, batch, pbar=None):
    nt, h, w = data.imgs.shape
    if data.mask is None:
        nx = h * w
    else:
        nx = np.count_nonzero(data.mask)

    dynamics = get_dynamics(dynamics)
    penalty = get_penalty(penalty)

    nk = yval.shape[0]
    if kind == "temporal":
        if data.mask is None:
            yval = yval.reshape(nk, nx)
        else:
            yval = yval[:, data.mask]
        bx = penalty.bt
        by = penalty.bx
        trans = True
        pena = penalty.la(yval)
    else:
        bx = penalty.bx
        by = penalty.bt
        trans = False
        pena = penalty.lu(yval)
        yval = dynamics(yval)

    ycor = matmul_batch(yval, data, trans, batch, pbar)
    yval, ycov, yout = calc_cov_out(yval)

    cx = 1 - jnp.square(bx)
    cy = 1 - jnp.square(by)

    a = ycor
    b = ycov - cx * yout
    c = yout - cy * ycov

    return Prepare(nt, nx, nk, pena, a, b, c)
