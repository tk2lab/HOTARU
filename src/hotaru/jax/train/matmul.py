import jax.numpy as jnp
import numpy as np

from ...io.mask import mask_range
from ..filter.map import mapped_imgs


def matmul_batch(x, imgs, mask, avgx, avgt, std0, trans, batch, pbar=None):

    if trans:

        def prepare(start, end):
            return jnp.array(imgs[start:end], jnp.float32)

        def apply(imgs):
            return jnp.matmul(x, imgs[:, mask].T)

        def finish(out):
            return np.concatenate(out, axis=1)

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
    return mapped_imgs(nt, prepare, apply, finish, finish, batch, pbar.update)
