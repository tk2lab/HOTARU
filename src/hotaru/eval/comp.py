import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, NullFormatter

from .seg_out import get_segment


def get_pair(a1, a2, thr=0.1):
    n = a1.shape[0]
    m = a2.shape[0]
    a1 = a1.reshape(n, -1)
    a2 = a2.reshape(m, -1)
    n1 = a1 / np.sqrt((a1 ** 2).sum(axis=1, keepdims=True))
    n2 = a2 / np.sqrt((a2 ** 2).sum(axis=1, keepdims=True))
    cos = n1 @ n2.T
    pair = []
    n = 0
    while True:
        n += 1
        i, j = divmod(np.argmax(cos), m)
        print(n, i, j, cos[i, j])
        if cos[i, j] < thr:
            break
        pair.append((i,j))
        cos[i, :] = 0
        cos[:, j] = 0
    return pair


def plot_comp_a(ax, pair, a1, a2, gamma=1.0, thr=0.6):
    n, h, w = a1.shape
    m = a2.shape[0]
    flag = ((1, 0, 0, 1),) * n + ((0, 0, 1, 1),) * m
    ax.imshow(get_segment(np.concatenate([a1, a2], axis=0), gamma, thr, flag))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0)


def plot_comp_v(ax, pair, time, v1, v2, cond=None, npair=None, n1=None, n2=None):
    if cond is None:
        cond = np.ones(time.size, bool)
    if cond is None:
        cond = np.on
    time = time[cond]
    v1mean = v1.mean(axis=1)
    v1min = v1.min(axis=1)
    v1max = v1.max(axis=1)
    v1 = (v1[:, cond] - v1mean[:, None]) / (v1max - v1min)[:, None]
    v2mean = v2.mean(axis=1)
    v2min = v2.min(axis=1)
    v2max = v2.max(axis=1)
    v2 = (v2[:, cond] - v2mean[:, None]) / (v2max - v2min)[:, None]
    n = v1.shape[0]
    m = v2.shape[0]
    k = 0
    num = 0
    for i in range(n):
        if i not in list(zip(*pair))[0]:
            a = (v1[i] - v1[i].mean()) / (v1[i].max() - v1[i].min())
            print(time.shape)
            print(a.shape)
            ax.plot(time, 2*a+k, c='r', alpha=0.5)
            k += 1
            num += 1
            if (n1 is not None) and (num >= n1):
                break
    for i, j in (pair if npair is None else pair[:npair]):
        a = (v1[i] - v1[i].mean()) / (v1[i].max() - v1[i].min())
        b = (v2[j] - v2[j].mean()) / (v2[j].max() - v2[j].min())
        ax.plot(time, 2*a+k, c='r', alpha=0.5)
        ax.plot(time, 2*b+k, c='b', alpha=0.5)
        k += 1
    num = 0
    for j in range(m):
        if j not in list(zip(*pair))[1]:
            b = (v2[j] - v2[j].mean()) / (v2[j].max() - v2[j].min())
            ax.plot(time, 2*b+k, c='b', alpha=0.5)
            k += 1
            num += 1
            if (n2 is not None) and (num >= n2):
                break
    ax.set_xlabel('time (sec)')
    ax.set_xlim(time[0], time[-1])
    ax.set_yticks([])
    ax.set_ylabel('cells')
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0)
