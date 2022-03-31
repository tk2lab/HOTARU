import click

from hotaru.footprint.find import find_peak

from .base import run_base
from .radius import radius_options
from .radius import radius_wrap


@click.command()
@radius_options
@click.option('--shard', type=int, default=1, show_default=True)
@click.option('--batch', type=int, default=100, show_default=True)
@radius_wrap
@run_base
def find(obj, radius, shard, batch):
    '''Find'''

    obj.stage = None

    data = obj.data()
    mask = obj.mask()
    avgx = obj.avgx()
    avgx[:] = 0.0
    nt = obj.nt()

    peaks = find_peak(
        data, mask, avgx, radius, shard, batch, nt, obj.verbose,
    )

    obj.save_csv(peaks, 'peak', stage='_find')
    return dict(npeaks=peaks.shape[0])
