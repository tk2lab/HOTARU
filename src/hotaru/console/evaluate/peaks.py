import click
import matplotlib.pyplot as plt

from ...evaluate.peaks import (
    plot_circle,
    plot_radius,
)
from ..base import configure


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--stage", type=int)
@click.option("--scale", type=float)
@click.pass_context
def peak(ctx, tag, stage, scale):
    """Visualize Statistics for Cleaned Candidates."""

    obj = ctx.obj
    data_tag = obj.log("3segment", tag, stage)["data_tag"]

    peak1 = obj.peaks(tag, stage)
    peak2 = peak1.query("accept == 'yes'")

    h, w = obj.mask(data_tag).shape
    radius_args = obj.used_radius_args(tag, stage)
    radius_min = radius_args["radius_min"]
    radius_max = radius_args["radius_max"]
    radius = obj.get_radius(**radius_args)

    fig = plt.figure(figsize=(1, 1))

    ax = fig.add_axes([3.8, 0.5, 3, 2])
    plot_radius(
        ax,
        peak1,
        "firmness",
        radius,
        hue="accept",
        palette=dict(yes="b", no="r"),
        alpha=0.5,
    )

    ax = fig.add_axes([0, -7 * h / w, 7, 7 * h / w])
    plot_circle(ax, peak2, h, w, scale, "firmness")

    path = obj.out_path("figure", tag, f"_{stage:03}-peak")
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.1)
    click.echo(f"see {path}.pdf")
