import click

from hotaru.footprint.find import find_peak

from ..base import run_command
from .options import radius_options


@run_command(
    *radius_options(),
    click.Option(["--shard"], type=int),
    click.Option(["--batch"], type=int),
)
def find(obj):
    """Find"""

    data = obj.data
    mask = obj.mask
    nt = obj.nt

    total = (nt + obj.shard - 1) // obj.shard
    with click.progressbar(length=total, label="Find") as prog:
        with obj.strategy.scope():
            peaks = find_peak(
                data,
                mask,
                obj.radius,
                obj.shard,
                obj.batch,
                prog=prog,
            )

    obj.save_csv(peaks, "peak", obj.find_tag, "_find")
    click.echo(f"num: {peaks.shape[0]}")
    return dict(npeaks=peaks.shape[0])
