import click

from hotaru.footprint.reduce import reduce_peak_idx_mp
from hotaru.footprint.reduce import label_out_of_range
from hotaru.footprint.make import make_segment

from .base import run_command


@run_command(
    click.Option(['--distance'], type=float),
    click.Option(['--window'], type=int),
    click.Option(['--batch'], type=int),
)
def init(obj):
    '''Init'''

    peaks = obj.peak
    radius_min = obj.used_radius_min
    radius_max = obj.used_radius_max

    idx = reduce_peak_idx_mp(peaks, obj.distance, obj.window, obj.verbose)
    peaks = label_out_of_range(peaks.loc[idx], radius_min, radius_max)

    data = obj.data
    mask = obj.mask

    accept = peaks.query('accept == "yes"')
    segment = make_segment(data, mask, accept, obj.batch, obj.verbose)

    obj.save_csv(peaks, 'peak', obj.init_tag, '_000')
    obj.save_numpy(segment, 'segment', obj.init_tag, stage='_000')
    click.echo(f'num: {segment.shape[0]}')
    return dict(num_cell=segment.shape[0])
