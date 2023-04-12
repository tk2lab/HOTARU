import os

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ...utils.gpu import get_gpu_memory


def mapped_imgs(imgs, apply, aggregate, finish, scale, pbar=None):
    num_devices = os.environ.get("HOTARU_NUM_DEVICES", jax.local_device_count())
    buffer = os.environ.get("HOTARU_BUFFER", get_gpu_memory()[0]) * (2**20)

    nt, h, w = imgs.shape
    print(scale, nt, h, w)
    batch = max(1, int(buffer / (4 * h * w * scale)))

    def calc(start, end):
        batch = end - start
        if end == nt:
            nd = 1
        else:
            nd = num_devices
        clip = jnp.asarray(imgs[start:end], jnp.float32)
        clip = clip.reshape(nd, batch, h, w)
        t0 = jnp.arange(start, end).reshape(nd, batch)
        pout = jax.pmap(apply)(t0, clip)
        return aggregate(*pout)

    if pbar is not None:
        pbar.reset(total=nt)
    out = []
    for start in range(0, nt, num_devices * batch):
        end = min(nt, start + num_devices * batch)
        out.append(calc(start, end))
        if pbar is not None:
            pbar.update(end - start)
    return finish(*zip(*out))
