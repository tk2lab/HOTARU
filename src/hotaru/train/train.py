import jax
import jax.numpy as jnp
import numpy as np

from .common import (
    calc_cov_out,
    mask_fn,
)
from .dynamics import get_dynamics
from .optimizer import ProxOptimizer
from .penalty import get_penalty


def optimize_temporal(mat, dynamics, penalty, lr, steps, scale=20, num_devices=1):
    opt = gen_optimizer("temporal", mat, dynamics, penalty, lr, scale, num_devices)
    opt.fit(**steps)
    return opt.val[0]


def optimize_spatial(mat, dynamics, penalty, lr, steps, scale=20, num_devices=1):
    opt = gen_optimizer("spatial", mat, dynamics, penalty, lr, scale, num_devices)
    opt.fit(**steps)
    return opt.val[0]


def gen_optimizer(kind, factor, dynamics, penalty, lr, scale, num_devices=1):
    def _calc_err(x, i):
        if kind == "temporal":
            x = dynamics(x)
        xval, xcov, xout = calc_cov_out(x, i, num_devices)
        return nn + (b[i] * xcov).sum() + (c[i] * xout).sum() - 2 * (a[i] * xval).sum()

    def loss_fwd(x):
        err = jax.pmap(_calc_err, in_axes=(None, 0))(x, dev_id).sum()
        var = err / nm
        return jnp.log(var) / 2 + py, (x, err)

    def loss_bwd(res, g):
        x, err = res
        grad_err_fn = jax.pmap(jax.grad(_calc_err), in_axes=(None, 0))
        grad_err = grad_err_fn(x, dev_id).sum(axis=0)[:nk]
        return (g / 2 * grad_err / (err + nn),)

    @jax.custom_vjp
    def loss_fn(x):
        return loss_fwd(x)[0]

    loss_fn.defvjp(loss_fwd, loss_bwd)

    dynamics = get_dynamics(dynamics)
    penalty = get_penalty(penalty)
    nt, nx, nk, py, a, b, c = factor
    ntf = float(nt)
    nxf = float(nx)
    nn = ntf * nxf
    nm = nn + ntf + nxf
    py /= nm

    dev_id = jnp.arange(num_devices)
    a = mask_fn(a, num_devices)
    b = mask_fn(b, num_devices)
    c = mask_fn(c, num_devices)

    if kind == "temporal":
        nu = nt + dynamics.size - 1
        init = [np.zeros((nk, nu))]
        pena = [penalty.lu]
    else:
        init = [np.zeros((nk, nx))]
        pena = [penalty.la]

    opt = ProxOptimizer(loss_fn, init, pena, num_devices, f"{kind} optimize")
    opt.set_params(lr / b.diagonal().max(), scale, nm)
    return opt
