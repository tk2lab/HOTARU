import click
import matplotlib.pyplot as plt

from ...evaluate.peaks import plot_circle
from ...evaluate.peaks import plot_radius
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

    find_tag = obj.get_config("make", make_tag, "find_tag")
    data_tag = obj.get_config("find", find_tag, "data_tag")

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

    cond = (peak1["radius"] > radius_min) & (peak1["radius"] < radius_max)
    peak2 = peak1[cond].copy()

    ymax = 1.1 * peak0.intensity.max()

    fig = plt.figure(figsize=(1, 1))

    ax = fig.add_axes([0.6, 0.5, 3, 2])
    plot_radius(
        ax,
        peak0,
        "intensity",
        radius,
        color="b",
        alpha=0.1,
    )
    ax.set_ylim(0, ymax)

    ax = fig.add_axes([3.8, 0.5, 3, 2])
    plot_radius(
        ax,
        peak1,
        "intensity",
        radius,
        hue="accept",
        palette=dict(yes="b", no="r"),
        alpha=0.5,
        ylabel=False,
    )
    ax.set_ylim(0, ymax)

    ax = fig.add_axes([0, -7 * h / w, 7, 7 * h / w])
    plot_circle(
        ax,
        peak2,
        h,
        w,
        distance,
        color="g",
    )

    path = obj.out_path("figure", make_tag, "_tune")
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.1)
    click.echo(f"see {path}.pdf")
