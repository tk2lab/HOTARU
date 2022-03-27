import numpy as np
import matplotlib.pyplot as plt


def plot_circle(ax, df, h, w, dist, **args):
    m = df['intensity'].max()
    for x, y, r, g in df[['x', 'y', 'radius', 'intensity']].values:
        ax.add_artist(plt.Circle((x, y), dist * r, alpha=g / m, fill=False, **args))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
