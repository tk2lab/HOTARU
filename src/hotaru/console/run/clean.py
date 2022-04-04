import numpy as np
import pandas as pd
import click

from hotaru.footprint.clean import modify_footprint
from hotaru.footprint.clean import clean_footprint
from hotaru.footprint.clean import check_accept

from .base import run_command
from .options import radius_options


@run_command(
    click.Option(['--stage', '-s'], type=int),
    click.Option(['--footprint-tag']),
    click.Option(['--footprint-stage'], type=int),
    *radius_options(),
    click.Option(['--thr-area-abs'], type=click.FloatRange(0.0)),
    click.Option(['--thr-area-rel'], type=click.FloatRange(0.0)),
    click.Option(['--thr-sim'], type=click.FloatRange(0.0, 1.0)),
    click.Option(['--batch'], type=click.IntRange(0))
)
def clean(obj):
    '''Clean'''

    if obj.stage == -1:
        obj['stage'] = '_curr'

    if obj.footprint_tag == '':
        if obj.stage == 1:
            obj['segment_tag'] = obj.init_tag
        else:
            obj['segment_tag'] = obj.tag
        obj['footprint_tag'] = obj.tag

    if obj.footprint_stage == -1:
        if isinstance(obj.stage, int):
            obj['segment_stage'] = obj.stage - 1
            obj['footprint_stage'] = obj.stage - 1
        else:
            obj['segment_stage'] = obj.stage
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
    obj.save_csv(peaks, 'peak')

    peaks = peaks.loc[index]
    cond = peaks['accept'] == 'yes'
    obj.save_numpy(footprint[~cond], 'removed')
    peaks.sort_values('firmness', ascending=False, inplace=True)

    old_nk = footprint.shape[0]
    nk = cond.sum()
    sim = peaks.loc[cond, 'sim'].values
    click.echo(peaks.loc[cond])
    click.echo(peaks.loc[~cond])
    click.echo(f'sim: {np.sort(sim[sim > 0])}')
    click.echo(f'ncell: {old_nk} -> {nk}')

    return dict(num_cell=nk)
