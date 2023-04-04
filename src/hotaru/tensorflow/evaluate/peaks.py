import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import (
    NullFormatter,
    ScalarFormatter,
)


def plot_circle(
    model=None,
    df=None,
    h=None,
    w=None,
    scale=1.0,
    g=None,
    ax=None,
    figwidth=5.0,
    **args
):
    if df is None:
        df = model.info

    if h is None:
        h = model.mask.shape[0]
    if w is None:
        w = model.mask.shape[1]

    if g is None:
        if "firmness" in df:
            g = "firmness"
        else:
            g = "intensity"

    if ax is None:
        fig = plt.figure(figsize=(figwidth, figwidth * h / w))
        ax = plt.gca()

    args.setdefault("edgecolor", "w")
    args.setdefault("alpha", 0.5)
    cmap = cm.get_cmap("Greens")
    gmax = df[g].max()

    ax.scatter(df.x, df.y, s=5, c="r")
    for x, y, r, k, g in df[["x", "y", "radius", "kind", g]].values:
        circle = plt.Circle(
            (x, y),
            scale * r,
            facecolor=cmap(g / gmax) if k == "cell" else "gray",
            **args,
        )
        ax.add_artist(circle)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)


def plot_radius(
    model=None, df=None, g=None, radius=None, xlabel=True, ylabel=True, ax=None, **args
):
    args.setdefault("s", 10)
    args.setdefault("edgecolor", "none")
    args.setdefault("rasterized", True)

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    if df is None:
        df = model.info

    if g is None:
        if "firmness" in df:
            g = "firmness"
        else:
            g = "intensity"

    if radius is None:
        radius = model.radius

    rmin = radius[0]
    rmax = radius[-1]
    logr = np.log(radius)
    diff = logr[1] - logr[0]
    rlist = (
        [rmin] + list(2 ** np.arange(np.log2(rmin), 1.1 * np.log2(rmax), 1.0)) + [rmax]
    )
    df["rj"] = np.exp(
        np.log(df.radius) + 0.8 * diff * (np.random.random(df.radius.size) - 0.5)
    )

    if "kind" in df:
        args["hue"] = "kind"
    sns.scatterplot(x="rj", y=g, data=df, ax=ax, **args)
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
