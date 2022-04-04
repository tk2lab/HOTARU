import numpy as np
from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import label, binary_closing
from scipy.ndimage import generate_binary_structure, binary_dilation
from matplotlib import cm

_reds = cm.get_cmap('Reds')
_blues = cm.get_cmap('Blues')
_greens = cm.get_cmap('Greens')

def get_segment(val, gauss, thr_out, color=None, mask=None):
    struct = generate_binary_structure(2, 2)
    if mask is None:
        n, h, w = val.shape
    else:
        h, w = mask.shape
        n = val.shape[0]
    if color is None:
        color = [(0, 1, 0, 1) for i in range(n)]
    out = np.zeros((h, w))
    bo = {c: False for c in color}
    for c, v in zip(color[::-1], val[::-1]):
        if mask is None:
            img = v
        else:
            img = np.zeros((h, w))
            img[mask] = v
        g = gaussian(img, gauss)
        g /= g.max()
        peak = np.argmax(g)
        lbl, n = label(g > thr_out)
        size = 0
        for i in range(1, n+1):
            tmp = binary_closing(lbl == i)
            ts = np.count_nonzero(tmp)
            if ts > size:
                size = ts
                obj = tmp
        out[obj] += 1.0
        bound = binary_dilation(obj, struct) ^ obj
        bo[c] |= bound
    out = out / out.max()
    out = _greens(out)
    for c, b in bo.items():
        out[b] = c
    return out
