import jax.numpy as jnp

from ..filter.map import mapped_imgs


def prepare(val, imgs, mask, avgx, avgt, std0, bx, by, trans, batch, pbar=None):
    cx = 1 - jnp.square(bx)
    cy = 1 - jnp.square(by)
    cor = matmul(val, imgs, stats, trans, buffer, num_devices, pbar)
    yval, ycov, yout = calc_cov_out(val)
    dat = -2 * cor
    cov = ycov - cx * yout
    out = yout - cy * ycov
    return dat, cov, out


def calc_cov_out(xval):
    nx = xval.shape[1]
    xcov = xval @ xval.T
    xsum = xval.sum(axis=1)
    xout = xsum[:, None] * (xsum / nx)
    return xval, xcov, xout


def matmul(x, imgs, mask, avgx, avgt, std0, trans, batch, pbar=None):

    if trans:

        def prepare(start, end):
            return jnp.array(imgs[start:end], jnp.float32)

        def apply(imgs):
            return jnp.matmul(x, imgs[:, mask].T)

        def finish(out):
            return jnp.concatenate(out, axis=1)

    else:

        def prepare(start, end):
            return (
                jnp.array(x[start:end], jnp.float32),
                jnp.array(imgs[start:end], jnp.float32),
            )

        def apply(x, imgs):
            return jnp.matmul(x, imgs[:, mask])

        def finish(out):
            return out.sum(axis=0)

    x0, y0, w, h = mask_range(mask)
    nt = imgs.shape[0]
    imgs = imgs[:, y0 : y0 + h, x0 : x0 + w]
    mask = mask[y0 : y0 + h, x0 : x0 + w]

    if pbar is not None:
        pbar = pbar(total=nt)
    return mapped_imgs(nt, prepare, apply, finish, finish, batch, pbar)
