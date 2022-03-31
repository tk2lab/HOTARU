import click

import pandas as pd

from hotaru.footprint.reduce import reduce_peak_idx_mp
from hotaru.footprint.reduce import label_out_of_range
from hotaru.footprint.make import make_segment

from .base import run_base


@click.command()
@click.option('--distance', type=float, default=1.6, show_default=True)
@click.option('--window', type=int, default=100, show_default=True)
@click.option('--batch', type=int, default=100, show_default=True)
@run_base
def init(obj, distance, window, batch):
    '''Init'''

    obj.stage = None

    peaks = obj.peak(initial=True)
    radius_min = obj.radius_min()
    radius_max = obj.radius_max()

    idx = reduce_peak_idx_mp(peaks, distance, window, obj.verbose)
    peaks = label_out_of_range(peaks.loc[idx], radius_min, radius_max)

    data = obj.data()
    mask = obj.mask()

    accept = peaks.query('accept == "yes"')
    segment, ng = make_segment(data, mask, accept, batch, obj.verbose)

    peaks.loc[ng, 'accept'] = 'no_seg'
    obj.save_csv(peaks, 'peak', stage='_init')
    obj.save_numpy(segment, 'segment', stage='_init')
    return dict()
