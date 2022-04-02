import numpy as np
from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_closing
from scipy.ndimage import label
from scipy.ndimage import binary_dilation


def calc_sim_area(segment, mask, gauss, thr_sim_area):
    nk = segment.shape[0]
    h, w = mask.shape
    seg = np.zeros((nk, h, w), np.float32)
    seg[:, mask] = segment
    if gauss > 0.0:
        seg = gaussian(seg, gauss)
    seg -= seg.min(axis=(1, 2), keepdims=True)
    mag = seg.max(axis=(1, 2))
    cond = mag > 0.0
    seg[cond] /= mag[cond, None, None]
    seg[~cond] = 1.0
    seg = seg > thr_sim_area
    cor = np.zeros((nk,))
    for j in np.arange(nk)[::-1]:
        cj = 0.0
        for i in np.arange(j):
            ni = np.count_nonzero(seg[i])
            nij = np.count_nonzero(seg[i] & seg[j])
            cij = nij / ni
            if cij > cj:
                cj = cij
        cor[j] = cj
    return cor


def calc_sim_cos(segment):
    nk = segment.shape[0]
    scale = np.sqrt((segment ** 2).sum(axis=1))
    seg = segment / scale[:, None]
    cor = np.zeros((nk,))
    for j in np.arange(nk)[::-1]:
        cj = 0.0
        for i in np.arange(j)[::-1]:
            cij = np.dot(seg[i], seg[j])
            if cij > cj:
                cj = cij
        cor[j] = cj
    return cor
