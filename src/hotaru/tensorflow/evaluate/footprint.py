import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
)
from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import (
    generate_binary_structure,
    label,
)


def plot_maximum(a, b=None, mask=None, vmin=0.0, vmax=1.0, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if mask is not None:
        h, w = mask.shape
        n = a.shape[0]
        val = a
        a = np.zeros((n, h, w))
        a[:, mask] = val
    else:
        h, w = a.shape[1:]
    ax.imshow(a.max(axis=0), cmap="Greens", vmin=vmin, vmax=vmax, alpha=0.7)
    if b is not None:
        if mask is not None:
            n = b.shape[0]
            val = b
            b = np.zeros((n, h, w))
            b[:, mask] = val
        ax.imshow(b.max(axis=0), cmap="Reds", vmin=vmin, vmax=vmax, alpha=0.3)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_linewidth(0)
    ax.spines["bottom"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)
    ax.spines["left"].set_linewidth(0)


def plot_contour(a, gauss=2.0, thr_out=0.5, cmap=None, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if cmap is None:
        cmap = plt.get_cmap("Greens")
    n, h, w = a.shape
    out, bound, _, _ = footprint_contour(a, gauss=gauss, thr_out=thr_out)
    out = cmap(out)
    out[bound[0]] = (0, 0, 1, 1)
    ax.imshow(out, cmap="Greens")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])


def footprint_contour(val, gauss, thr_out, mask=None, kind=None):
    if kind is None:
        kind = np.ones(val.shape[0], bool)
    struct = generate_binary_structure(2, 2)
    if mask is None:
        n, h, w = val.shape
    else:
        h, w = mask.shape
    out = 0
    area = []
    mean = []
    uni, ind = np.unique(kind, return_inverse=True)
    b = np.zeros((len(uni), h, w), bool)
    for k, v in zip(ind[::-1], val[::-1]):
        if mask is None:
            img = v
        else:
            img = np.zeros((h, w))
            img[mask] = v
        if gauss > 0.0:
            g = gaussian(img, gauss)
        else:
            g = img
        g /= g.max()
        lbl, n = label(g > thr_out)
        obj = np.zeros_like(lbl, bool)
        size = 0
        for i in range(1, n + 1):
            tmp_obj = binary_closing(lbl == i)
            tmp_size = np.count_nonzero(tmp_obj)
            if tmp_size > size:
                obj = tmp_obj
                size = tmp_size
        if size > 0:
            out += obj
            bound = binary_dilation(obj, struct) ^ obj
            b[k] |= bound
        area.append(size)
        mean.append(img[obj].mean() if size > 0 else 0.0)
    return out, b, np.array(area)[::-1], np.array(mean)[::-1]
