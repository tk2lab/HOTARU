import click

from ...footprint.find import find_peak
from ...util.progress import Progress
from ..base import (
    command_wrap,
    configure,
    radius_options,
)


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--data-tag", type=str)
@radius_options
@click.option("--shard", type=int)
@click.option("--batch", type=int)
@click.option("--threshold-region", type=float)
@click.pass_obj
@command_wrap
def find(obj, tag, data_tag, radius, shard, batch, threshold_region):
    """Find Initial Cell Candidate Peaks."""

    if data_tag is None:
        data_tag = tag

    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    nt = obj.nt(data_tag)

    total = (nt + shard - 1) // shard
    with Progress(length=total, label="Find", unit="frame") as prog:
        with obj.strategy.scope():
            peaks = find_peak(
                data, mask, radius, shard, batch, threshold_region, prog=prog
            )
    nk = peaks.shape[0]
    click.echo(f"num: {nk}")
    obj.save_csv(peaks, tag, 0, "2find", "info")

    return dict(data_tag=data_tag, nk=nk), tag, 0, "2find"
