import os

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ...utils.gpu import get_gpu_memory


def mapped_imgs(imgs, apply, aggregate, finish, scale, pbar=None):
    nd = os.environ.get("HOTARU_NUM_DEVICES", jax.local_device_count())
    buffer = os.environ.get("HOTARU_BUFFER", get_gpu_memory()[0]) * (2**20)

    nt, h, w = imgs.shape
    batch = max(1, int(buffer / (4 * h * w * scale)))

    def calc(start, end):
        clip = jnp.asarray(imgs[start:end], jnp.float32)
        frame = jnp.arange(start, end)
        size = end - start
        if end == nt:
            clip = jnp.pad(clip, [[0, nd * batch - size], [0, 0], [0, 0]], constant_values=jnp.nan)
            frame = jnp.pad(frame, [[0, nd * batch - size]], constant_values=-1)
        clip = clip.reshape(nd, batch, h, w)
        frame = frame.reshape(nd, batch)
        pout = jax.pmap(apply)(frame, clip)
        return aggregate(*(np.array(p.block_until_ready()) for p in pout))

    if pbar is not None:
        pbar.reset(total=nt)
    out = []
    for start in range(0, nt, nd * batch):
        end = min(nt, start + nd * batch)
        out.append(calc(start, end))
        if pbar is not None:
            pbar.update(end - start)
    return finish(*zip(*out))
