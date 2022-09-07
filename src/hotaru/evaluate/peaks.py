import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm


def plot_circle(ax, df, h, w, scale, **args):
    args.setdefault("edgecolor", "w")
    args.setdefault("alpha", 0.5)
    cmap = cm.get_cmap("Greens")
    gmax = df.intensity.max()

    ax.scatter(df.x, df.y, s=5, c="r")
    for x, y, r, g in df[["x", "y", "radius", "intensity"]].values:
        circle = plt.Circle(
            (x, y), scale * r, facecolor=cmap(g / gmax), **args,
        )
        ax.add_artist(circle)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)


def plot_radius(ax, df, yname, radius, xlabel=True, ylabel=True, **args):
    args.setdefault("s", 10)
    args.setdefault("edgecolor", "none")
    args.setdefault("rasterized", True)

    rmin = radius[0]
    rmax = radius[-1]
    logr = np.log(radius)
    diff = logr[1] - logr[0]
    rlist = (
        [rmin]
        + list(2 ** np.arange(np.log2(rmin), 1.1 * np.log2(rmax), 1.0))
        + [rmax]
    )
    df["rj"] = np.exp(
        np.log(df.radius)
        + 0.8 * diff * (np.random.random(df.radius.size) - 0.5)
    )

    sns.scatterplot(x="rj", y=yname, data=df, ax=ax, **args)
    ax.set_xscale("log")
    ax.set_xticks(rlist[1:-1])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(rlist[0] * 0.8, rlist[-1] * 1.2)
    if xlabel:
        ax.set_xlabel("radius (pixel)")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    if not ylabel:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)
