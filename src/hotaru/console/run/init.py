import click

from hotaru.footprint.reduce import reduce_peak_idx_mp
from hotaru.footprint.reduce import label_out_of_range
from hotaru.footprint.make import make_segment

from .base import run_base


@click.command()
@click.option('--find-tag', '-F', show_default='auto')
@click.option('--distance', type=float, default=1.6, show_default=True)
@click.option('--window', type=int, default=100, show_default=True)
@click.option('--batch', type=int, default=100, show_default=True)
@run_base
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

    obj.save_csv(peaks, 'peak', stage='_000')
    obj.save_numpy(segment, 'segment', stage='_000')
    return dict()
