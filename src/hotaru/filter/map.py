import jax
import jax.numpy as jnp
import numpy as np

from ..utils import get_progress


def mapped_imgs(nt, prepare, apply, types, batch, pbar=None):

    def init():
        out = [None] * len(types)
        for i, (k, t) in enumerate(types):
            match k:
                case "min":
                    out[i] = jnp.zeros((), t)
                case "max":
                    out[i] = jnp.zeros((), t)
                case "add":
                    out[i] = jnp.zeros((), t)
                case "stack":
                    out[i] = []
                case "argmax":
                    v = jnp.nan if t == jnp.float32 else -1
                    out[i] = jnp.full((), v, dtype=t)
        return out

    def calc(start, end, out):
        clip = prepare(start, end)
        if not isinstance(clip, tuple):
            clip = (clip,)

        diff = batch_size - (end - start)
        if diff != 0:

            def pad(c):
                return jnp.pad(
                    c,
                    [[0, diff]] + [[0, 0]] * (c.ndim - 1),
                    constant_values=jnp.nan if c.dtype == jnp.float32 else -1,
                )

            clip = map(pad, clip)
        clip = [c.reshape(batch + c.shape[1:]) for c in clip]

        val = jax.pmap(apply)(*clip)
        if not isinstance(val, tuple):
            val = (val,)

        for i, (k, t) in enumerate(types):
            match k:
                case "min":
                    vali = val[i].min(axis=0)
                    out[i] = jnp.minimum(out[i], vali)
                case "max":
                    vali = val[i].max(axis=0)
                    out[i] = jnp.maximum(out[i], vali)
                case "add":
                    vali = val[i].sum(axis=0)
                    out[i] += vali
                case "stack":
                    s0, s1, *shape = val[i].shape
                    vali = val[i].reshape([s0 * s1] + shape)
                    out[i].append(vali)
                case "argmax":
                    idx = jnp.argmax(val[0], axis=0, keepdims=True)
                    val0 = jnp.take_along_axis(val[0], idx, axis=0)[0]
                    vali = jnp.take_along_axis(val[i], idx, axis=0)[0]
                    out[i] = jnp.where(out[0] > val0, out[i], vali)
        return out

    def finish(out):
        for i, (k, t) in enumerate(types):
            match k:
                case "stack":
                    out[i] = jnp.concatenate(out[i], axis=0)[:nt]
                case _:
                    pass
            out[i] = np.array(out[i])
        return out

    pbar = get_progress(pbar)
    batch = tuple(batch)
    batch_size = np.prod(batch)
    out = init()
    for start in range(0, nt, batch_size):
        end = min(nt, start + batch_size)
        out = calc(start, end, out)
        pbar.update(end - start)
    return finish(out)
