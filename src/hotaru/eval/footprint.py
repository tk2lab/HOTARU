import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_closing
from scipy.ndimage import binary_dilation


def calc_sim_cos(segment):
    nk = segment.shape[0]
    segment = segment.reshape(nk, -1)
    scale = np.sqrt((segment ** 2).sum(axis=1))
    seg = segment / scale[:, None]
    cor = seg @ seg.T
    max_cor = np.zeros(nk)
    for j in np.arange(1, nk)[::-1]:
        max_cor[j] = cor[j, :j].max()
    return max_cor


def calc_sim_area(segment, mask=None):
    nk = segment.shape[0]
    segment = segment.reshape(nk, -1).astype(np.float32)
    scale = segment.sum(axis=1)
    cor = (segment @ segment.T) / scale
    max_cor = np.zeros(nk)
    for j in np.arange(1, nk)[::-1]:
        if mask is None:
            max_cor[j] = cor[j, :j].max()
        elif np.any(mask[:j]):
            max_cor[j] = cor[j, :j][mask[:j]].max()
    return max_cor


def plot_maximum(ax, a, vmin=0.0, vmax=1.0):
    n, h, w = a.shape
    ax.imshow(a.max(axis=0), cmap='Greens', vmin=vmin, vmax=vmax)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0)


def plot_contour(ax, a, gauss=2.0, thr_out=0.1):
    n, h, w = a.shape
    out, bound, _, _ = footprint_contour(a, gauss=2.0, thr_out=0.1)
    out = plt.get_cmap('Greens')(out)
    out[bound[0]] = (0, 0, 1, 1)
    ax.imshow(out, cmap='Greens')
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
        for i in range(1, n+1):
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
