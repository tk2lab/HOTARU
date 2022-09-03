import click

from hotaru.footprint.reduce import label_out_of_range
from hotaru.footprint.reduce import reduce_peak_idx_data
from hotaru.footprint.reduce import reduce_peak_idx_finish

from ..base import run_command


@run_command(
    click.Option(["--distance"], type=float),
    click.Option(["--window"], type=int),
)
def test(obj):
    """Test"""

    peaks = obj.peak
    radius_min = obj.used_radius_min
    radius_max = obj.used_radius_max

    data = reduce_peak_idx_data(peaks, obj.distance, obj.window)
    with click.progressbar(iterable=data, label="Reduce") as data:
        idx = reduce_peak_idx_finish(data)
    peaks = label_out_of_range(peaks.loc[idx], radius_min, radius_max)
    obj.save_csv(peaks, "peak", obj.init_tag, "")
    nk = peaks.query("accept == 'yes'")
    click.echo(f"num: {nk}")
    return dict(num_cell=nk)
