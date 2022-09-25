import click
import numpy as np

from ...footprint.make import make_segment
from ...footprint.reduce import label_out_of_range
from ...footprint.reduce import reduce_peak_idx_data
from ...footprint.reduce import reduce_peak_idx_finish
from ..base import command_wrap
from ..base import configure
from ..progress import Progress


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--find-tag", type=str)
@click.option("--distance", type=float)
@click.option("--window", type=int)
@click.option("--batch", type=int)
@click.option("--only-reduce", is_flag=True)
@click.pass_obj
@command_wrap
def make(obj, tag, find_tag, distance, window, batch, only_reduce):
    """Make Initial Segment."""

    peaks = obj.peaks(find_tag)
    idx_data = reduce_peak_idx_data(peaks, distance, window)
    with Progress(iterable=idx_data, label="Reduce", unit="block") as prog:
        idx = reduce_peak_idx_finish(prog)

    radius = obj.used_radius(find_tag)
    peaks = label_out_of_range(peaks.loc[idx], radius)
    obj.save_csv(peaks, "peak", tag, 1)
    cell = peaks.kind == "cell"
    local = peaks.kind == "local"
    click.echo(f"num: {cell.sum()}, {local.sum()}")

    if only_reduce:
        return {}, "3segment", tag, "tune"

    data_tag = obj.data_tag("2find", find_tag, 0)
    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    nt = obj.nt(data_tag)

    with Progress(length=nt, label="Make", unit="frame") as prog:
        with obj.strategy.scope():
            peaks = peaks[cell | local]
            segment = make_segment(data, mask, peaks, batch, prog=prog)
    cell = peaks.kind == "cell"
    local = peaks.kind == "local"
    obj.save_numpy(segment[cell], "segment", tag, 1)
    obj.save_numpy(segment[local], "localx", tag, 1)

    log = dict(data_tag=data_tag)
    return log, "2segment", tag, 1
