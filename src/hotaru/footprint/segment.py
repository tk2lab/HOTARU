import jax.lax as lax
import jax.numpy as jnp


def get_segment_mask(val, y0, x0):
    def cond(args):
        seg, old_seg = args
        return jnp.any(seg != old_seg)

    def body(args):
        seg, old_seg = args
        old_seg = seg.copy()
        for dy, dx in delta:
            nseg = jnp.roll(seg, (dy, dx), axis=(0, 1))
            if dy == 1:
                nseg = nseg.at[0, :].set(False)
            if dy == -1:
                nseg = nseg.at[-1, :].set(False)
            if dx == 1:
                nseg = nseg.at[:, 0].set(False)
            if dx == -1:
                nseg = nseg.at[:, -1].set(False)
            nval = jnp.roll(val, (dy, dx), axis=(0, 1))
            nseg &= (0 < val) & (val <= nval)
            seg |= nseg
        return seg, old_seg

    delta = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if (dy, dx) != (0, 0)]

    old_seg = jnp.zeros_like(val, bool)
    seg = jnp.zeros_like(val, bool)
    seg = seg.at[y0, x0].set(True)
    init = seg, old_seg
    return lax.while_loop(cond, body, init)[0]
