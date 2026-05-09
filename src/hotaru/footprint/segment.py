from keras import ops


def get_segment_mask(val, y0, x0):
    delta = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if (dy, dx) != (0, 0)]
    h, w = val.shape

    def cond(seg, old_seg):
        return ops.any(seg != old_seg)

    def body(seg, _old_seg):
        old_seg = seg.copy()
        for dy, dx in delta:
            nseg = ops.roll(seg, (dy, dx), axis=(0, 1))
            if dy == 1:
                nseg = ops.slice_update(nseg, (0, 0), ops.zeros((1, w), 'uint8'))
            if dy == -1:
                nseg = ops.slice_update(nseg, (-1, 0), ops.zeros((1, w), 'uint8'))
            if dx == 1:
                nseg = ops.slice_update(nseg, (0, 0), ops.zeros((h, 1), 'uint8'))
            if dx == -1:
                nseg = ops.slice_update(nseg, (0, -1), ops.zeros((h, 1), 'uint8'))
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
