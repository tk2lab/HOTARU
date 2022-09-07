import click

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

    data_tag = obj.log("2find", find_tag, 0)["data_tag"]
    data = obj.data(data_tag)
    mask = obj.mask(data_tag)
    nt = obj.nt(data_tag)

    peaks = obj.peaks(find_tag)
    radius_arg = obj.used_radius_args(find_tag)
    radius_min = radius_arg["radius_min"]
    radius_max = radius_arg["radius_max"]

    idx_data = reduce_peak_idx_data(peaks, distance, window)
    with Progress(iterable=idx_data, label="Reduce", unit="block") as prog:
        idx = reduce_peak_idx_finish(prog)
    peaks = label_out_of_range(peaks.loc[idx], radius_min, radius_max)
    obj.save_csv(peaks, "peak", tag, 0)

    accept = peaks.query("accept == 'yes'")
    click.echo(f"num: {accept.shape[0]}")

    log = dict(data_tag=data_tag, find_tag=find_tag)

    if only_reduce:
        return log, "3segment", tag, "tune"
    else:
        with Progress(length=nt, label="Make", unit="frame") as prog:
            with obj.strategy.scope():
                segment = make_segment(data, mask, accept, batch, prog=prog)
        obj.save_numpy(segment, "segment", tag, 0)
        return log, "3segment", tag, 0
