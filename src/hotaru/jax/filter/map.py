import os

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


def mapped_imgs(nt, prepare, apply, aggregate, finish, batch, callback=lambda n: None):

    def calc(start, end):
        clip = prepare(start, end)
        if not isinstance(clip, tuple):
            clip = (clip,)

        diff = batch_size - (end - start)
        if diff != 0:
            def pad(c):
                return jnp.pad(
                    c, [[0, diff]] + [[0, 0]] * (c.ndim - 1),
                    constant_values=np.nan if c.dtype == jnp.float32 else -1,
                )
            clip = map(pad, clip)
        clip = [c.reshape(batch + c.shape[1:]) for c in clip]

        pout = jax.pmap(apply)(*clip)
        if not isinstance(pout, tuple):
            pout = (pout,)

        def to_numpy(x):
            return np.array(x)
        return aggregate(*map(to_numpy, pout))

    batch = tuple(batch)
    batch_size = np.prod(batch)
    out = []
    for start in range(0, nt, batch_size):
        end = min(nt, start + batch_size)
        out.append(calc(start, end))
        callback(end - start)
    if not isinstance(out[0], tuple):
        return finish(out)
    return finish(*zip(*out))
