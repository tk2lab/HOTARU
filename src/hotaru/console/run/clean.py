import numpy as np
import pandas as pd
import click

from hotaru.footprint.clean import modify_footprint
from hotaru.footprint.clean import clean_footprint
from hotaru.footprint.clean import check_accept

from .base import run_base


@click.command()
@click.option('--thr-area-abs', type=float, default=np.inf, show_default=True)
@click.option('--thr-area-rel', type=float, default=0.0, show_default=True)
@click.option('--thr-sim', type=float, default=1.0, show_default=True)
@click.option('--initial', is_flag=True)
@click.option('--batch', type=int, default=100, show_default=True)
@run_base
def clean(obj, thr_area_abs, thr_area_rel, thr_sim, initial, batch):
    '''Clean'''

    if obj.prev_stage is None:
        if isinstance(obj.stage, int):
            obj.prev_stage = obj.stage - 1
        else:
            obj.prev_stage = obj.stage

    if obj.prev_stage == 0:
        initial = True

    mask = obj.mask()

    print(obj.stage)
    print(obj.prev_stage)
    footprint = obj.footprint()
    index = obj.index(initial)
    print(footprint.shape)
    print(len(index))

    cond = modify_footprint(footprint)
    no_seg = pd.DataFrame(index=index[~cond])
    no_seg['accept'] = 'no_seg'

    segment, peaks = clean_footprint(
        footprint[cond], index[cond],
        mask, obj.radius, batch, obj.verbose,
    )

    idx = np.argsort(peaks['firmness'].values)[::-1]
    segment = segment[idx]
    peaks = peaks.iloc[idx].copy()

    check_accept(
        segment, peaks, obj.radius,
        thr_area_abs, thr_area_rel, thr_sim,
    )

    cond = peaks['accept'] == 'yes'
    obj.save_numpy(segment[cond], 'segment')

    peaks = pd.concat([peaks, no_seg], axis=0)
    peaks = peaks.loc[index]
    cond = peaks['accept'] == 'yes'
    obj.save_numpy(footprint[~cond], 'removed')

    peaks.sort_values('firmness', ascending=False, inplace=True)
    obj.save_csv(peaks, 'peak')

    old_nk = footprint.shape[0]
    nk = cond.sum()
    sim = peaks.loc[cond, 'sim'].values
    click.echo(peaks.loc[cond])
    click.echo(peaks.loc[~cond])
    click.echo(f'sim: {np.sort(sim[sim > 0])}')
    click.echo(f'ncell: {old_nk} -> {nk}')
    return dict()
