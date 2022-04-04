import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, NullFormatter


def plot_radius(ax, df, yname, cname, radius, xlabel=True, ylabel=True, **args):
    rmin = radius[0]
    rmax = radius[-1]
    logr = np.log(radius)
    diff = logr[1] - logr[0]
    rlist = [rmin] + list(2 ** np.arange(np.log2(rmin), 1.1 * np.log2(rmax), 1.0)) + [rmax]

    r = df['radius']
    y = df[yname]
    df['rj'] = np.exp(np.log(r) + 0.8 * diff * (np.random.random(r.size) - 0.5))
    sns.scatterplot(x='rj', y=yname, hue=cname, data=df, ax=ax, **args)
    ax.set_xscale('log')
    ax.set_xticks(rlist[1:-1])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(rlist[0]*0.8, rlist[-1]*1.2)
    ax.set_ylim(0, y.max() * 1.1)
    if xlabel:
        ax.set_xlabel('radius (pixel)')
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])
    if not ylabel:
        ax.set_ylabel('')
        ax.set_yticklabels([])
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
