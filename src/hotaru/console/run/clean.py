import numpy as np
import pandas as pd
import click

from hotaru.footprint.clean import modify_footprint
from hotaru.footprint.clean import clean_footprint
from hotaru.footprint.clean import check_accept

from .base import run_base
from .options import radius_options


@click.command()
@click.option('--stage', '-s', type=int, show_default='no stage')
@click.option('--footprint-tag', '-T', show_default='auto')
@click.option('--footprint-stage', '-S', type=int, show_default='auto')
@radius_options
@click.option('--thr-area-abs', type=click.FloatRange(0.0, np.inf), default=np.inf, show_default=True)
@click.option('--thr-area-rel', type=click.FloatRange(0.0, np.inf), default=0.0, show_default=True)
@click.option('--thr-sim', type=click.FloatRange(0.0, 1.0))
@click.option('--batch', type=int, default=100, show_default=True)
@run_base
def clean(obj):
    '''Clean'''

    if obj.footprint_tag is None:
        obj['footprint_tag'] = obj.tag

    if obj.footprint_stage is None:
        if isinstance(obj.stage, int):
            obj['footprint_stage'] = obj.stage - 1
        else:
            obj['footprint_stage'] = obj.stage

    mask = obj.mask

    footprint = obj.footprint
    index = obj.index

    cond = modify_footprint(footprint)
    no_seg = pd.DataFrame(index=index[~cond])
    no_seg['accept'] = 'no_seg'

    segment, peaks = clean_footprint(
        footprint[cond], index[cond],
        mask, obj.radius, obj.batch, obj.verbose,
    )

    idx = np.argsort(peaks['firmness'].values)[::-1]
    segment = segment[idx]
    peaks = peaks.iloc[idx].copy()

    check_accept(
        segment, peaks, obj.radius,
        obj.thr_area_abs, obj.thr_area_rel, obj.thr_sim,
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
