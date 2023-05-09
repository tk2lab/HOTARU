import jax.numpy as jnp
import numpy as np

from ..filter.map import mapped_imgs


def matmul_batch(x, y, trans, batch, pbar=None):
    imgs, mask, avgx, avgt, std0 = y

    if trans:

        def prepare(start, end):
            return (
                jnp.array(imgs[start:end], jnp.float32),
                jnp.array(avgt[start:end], jnp.float32),
            )

        def apply(imgs, avgt):
            if mask is None:
                y = ((imgs - avgx).reshape(-1, h * w) - avgt[:, None]) / std0
            else:
                y = ((imgs - avgx)[:, mask] - avgt[:, None]) / std0
            return jnp.matmul(x, y.T)

        def finish(out):
            return np.concatenate(out, axis=1)

    else:

        def prepare(start, end):
            return (
                jnp.array(x.T[start:end], jnp.float32),
                jnp.array(imgs[start:end], jnp.float32),
                jnp.array(avgt[start:end], jnp.float32),
            )

        def apply(x, imgs, avgt):
            if mask is None:
                y = ((imgs - avgx).reshape(-1, h * w) - avgt[:, None]) / std0
            else:
                y = ((imgs - avgx)[:, mask] - avgt[:, None]) / std0
            return jnp.matmul(x.T, y)

        def finish(out):
            return np.sum(out, axis=0)

    nt, h, w = imgs.shape

    if pbar is not None:
        pbar = pbar(total=nt)
        pbar.set_description("matmul")
        pbar = pbar.update
    return mapped_imgs(nt, prepare, apply, finish, finish, batch, pbar)
