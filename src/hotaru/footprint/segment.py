from keras import ops


def get_segment_mask(val, y0, x0):
    delta = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if (dy, dx) != (0, 0)]
    zero = ops.zeros_like(val[0], 'uint8')

    def cond(args):
        seg, old_seg = args
        return ops.any(seg != old_seg)

    def body(args):
        seg, _old_seg = args
        old_seg = seg.copy()
        for dy, dx in delta:
            nseg = ops.roll(seg, (dy, dx), axis=(0, 1))
            if dy == 1:
                nseg = ops.slice_update(nseg, (0, 0), zero)
            if dy == -1:
                nseg = ops.slice_update(nseg, (-1, 0), zero)
                nseg = nseg.at[-1, :].set(False)
            if dx == 1:
                nseg = ops.slice_update(nseg, (0, 0), zero[:, None])
            if dx == -1:
                nseg = ops.slice_update(nseg, (0, -1), zero[:, None])
            nval = ops.roll(val, (dy, dx), axis=(0, 1))
            nseg &= (val > 0) & (val <= nval)
            seg |= nseg
        return seg, old_seg

    old_seg = ops.zeros_like(val, 'uint8')
    seg = ops.zeros_like(val, 'uint8')
    seg = ops.scatter_update(seg, (y0, x0), 1)
    init = seg, old_seg
    seg, _old_seg = ops.while_loop(cond, body, init)
    return seg
