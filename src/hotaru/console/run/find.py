import click

from hotaru.footprint.find import find_peak

from .base import run_base


@click.command()
@click.option('--shard', type=int, default=1, show_default=True)
@click.option('--batch', type=int, default=100, show_default=True)
@run_base
def find(obj, shard, batch):
    '''Find'''

    obj.stage = None

    data = obj.data()
    mask = obj.mask()
    nt = obj.nt()

    peaks = find_peak(
        data, mask, obj.radius, shard, batch, nt, obj.verbose,
    )

    obj.save_csv(peaks, 'peak', stage='_find')
    return dict(radius=obj.radius, npeaks=peaks.shape[0])
