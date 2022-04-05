import numpy as np
import matplotlib.pyplot as plt


def plot_circle(ax, df, h, w, dist, **args):
    m = df['intensity'].values
    df['g'] = 0.5 * ((m - m.min()) / (m.max() - m.min())) + 0.5
    for x, y, r, g in df[['x', 'y', 'radius', 'g']].values:
        ax.add_artist(plt.Circle((x, y), dist * r, alpha=g, fill=False, **args))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
