import numpy as np
import matplotlib.pyplot as plt


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
