import click

from ...footprint.make import make_segment
from ...footprint.reduce import label_out_of_range
from ...footprint.reduce import reduce_peak_idx_data
from ...footprint.reduce import reduce_peak_idx_finish
from ..base import run_command


@run_command(
    click.Option(["--distance"], type=float),
    click.Option(["--window"], type=int),
    click.Option(["--batch"], type=int),
)
def init(obj):
    """Init"""

    peaks = obj.peak
    radius_min = obj.used_radius_min
    radius_max = obj.used_radius_max

    data = reduce_peak_idx_data(peaks, obj.distance, obj.window)
    with click.progressbar(iterable=data, label="Reduce") as data:
        idx = reduce_peak_idx_finish(data)
    peaks = label_out_of_range(peaks.loc[idx], radius_min, radius_max)
    accept = peaks.query("accept == 'yes'")
    click.echo(f"num: {accept.shape[0]}")

    with click.progressbar(length=obj.nt, label="Make") as prog:
        with obj.strategy.scope():
            segment = make_segment(
                obj.data,
                obj.mask,
                accept,
                obj.batch,
                prog=prog,
            )
    obj.save_csv(peaks, "peak", obj.init_tag, "_000")
    obj.save_numpy(segment, "segment", obj.init_tag, stage="_000")
    return dict(num_cell=segment.shape[0])
