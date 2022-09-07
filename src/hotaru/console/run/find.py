import click

from ...footprint.find import find_peak
from ..base import command_wrap
from ..base import configure
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--data-tag", type=str)
@click.option("--radius-type", type=click.Choice(["log", "linear"]))
@click.option("--radius-min", type=float)
@click.option("--radius-max", type=float)
@click.option("--radius-num", type=int)
@click.option("--shard", type=int)
@click.option("--batch", type=int)
@click.pass_obj
@command_wrap
def find(obj, tag, data_tag, shard, batch, **radius_args):
    """Find Initial Cell Candidate Peaks."""

    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    nt = obj.nt(data_tag)
    radius = obj.get_radius(**radius_args)

    total = (nt + shard - 1) // shard
    with Progress(length=total, label="Find", unit="frame") as prog:
        with obj.strategy.scope():
            peaks = find_peak(data, mask, radius, shard, batch, prog=prog)
    num_cell = peaks.shape[0]
    click.echo(f"num: {num_cell}")
    obj.save_csv(peaks, "peak", tag, "-find")

    log = dict(data_tag=data_tag)
    return log, "2find", tag, 0
