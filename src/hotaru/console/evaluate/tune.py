import click
import matplotlib.pyplot as plt

from ...evaluate.circle import plot_circle
from ...evaluate.radius import plot_radius
from ..base import configure
from ..run.data import data
from ..run.find import find
from ..run.make import make


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--make-tag", type=str)
@click.pass_context
def tune(ctx, tag, make_tag):
    """Visualize Statistics for Tuning Initial Candidates."""

    obj = ctx.obj

    find_tag = obj.config.get(f"make/{make_tag}", "find_tag")
    data_tag = obj.config.get(f"find/{find_tag}", "data_tag")

    obj.invoke(ctx, data, f"--tag={data_tag}")
    obj.invoke(ctx, find, f"--tag={find_tag}")
    obj.invoke(ctx, make, f"--tag={make_tag}", "--only-reduce")

    radius_args = obj.used_radius_args(find_tag)
    radius_min = radius_args["radius_min"]
    radius_max = radius_args["radius_max"]
    radius = obj.get_radius(**radius_args)
    distance = obj.used_distance(make_tag)
    h, w = obj.mask(data_tag).shape

    peak0 = obj.peaks(find_tag)
    peak1 = obj.peaks(make_tag, 0)

    fig = plt.figure(figsize=(1, 1))
    cmap = dict(r="r", g="g", b="b")

    ax = fig.add_axes([0, 0, 3, 2])
    peak0["color"] = "b"
    plot_radius(
        ax,
        peak0,
        "intensity",
        "color",
        radius,
        palette=cmap,
        edgecolor="none",
        alpha=0.2,
        size=2,
        legend=False,
        rasterized=True,
    )

    ax = fig.add_axes([4, 0, 3, 2])
    plot_radius(
        ax,
        peak1,
        "intensity",
        "accept",
        radius,  # palette=cmap,
        edgecolor="none",
        alpha=0.2,
        size=2,
        legend=False,
        rasterized=True,
        ylabel=False,
    )

    ax = fig.add_axes([0, 3, 7, 7 * h / w])
    cond = (peak1["radius"] > radius_min) & (peak1["radius"] < radius_max)
    ax.scatter(
        peak1.loc[cond, "x"].values,
        peak1.loc[cond, "y"].values,
        s=5,
        c="r",
    )
    plot_circle(
        ax,
        peak1[cond].copy(),
        h,
        w,
        distance,
        color="g",
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    path = obj.out_path("figure", make_tag, "_tune")
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.1)
    click.echo(f"see {path}.pdf")
