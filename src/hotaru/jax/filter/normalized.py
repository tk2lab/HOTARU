import tqdm
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp


global_buffer = 2 ** 30


def apply_to_normalized(apply, finish, imgs, stats=None, batch=100, num_devices=None, pbar=None):
    if num_devices is None:
        num_devices = jax.local_device_count()

    if stats is None:
        nt, h, w = imgs.shape
        x0, y0 = 0, 0
        mask = np.ones((h, w), bool)
        avgx = jnp.zeros((h, w))
        avgt = jnp.zeros(nt)
        std0 = jnp.ones(())
    else:
        nt, x0, y0, mask, avgx, avgt, std0, min0, max0 = stats
        h, w = mask.shape

    imgs = imgs[:, y0:y0+h, x0:x0+w]

    def calc(imgs, avgt):
        return apply((imgs - avgx - avgt) / std0, mask)

    def pmap_calc(imgs, avgt):
        imgs = imgs.reshape(num_devices, batch, h, w)
        avgt = avgt.reshape(num_devices, batch, 1, 1)
        out = jax.pmap(calc)(imgs, avgt)
        return finish(out)

    end = 0
    while end < nt:
        start = end
        end = min(nt, start + num_devices * batch)
        batch = end - start
        if end == nt:
            num_devices = 1
        clip = jnp.asarray(imgs[start:end], jnp.float32)
        clipt = avgt[start:end]
        yield start, pmap_calc(clip, clipt)
        if pbar is not None:
            pbar.update(end - start)
