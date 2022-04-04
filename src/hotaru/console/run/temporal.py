import click

from hotaru.optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from hotaru.train.variance import Variance
from hotaru.train.spike import SpikeModel

from .base import run_base
from .options import model_options


@click.command()
@click.option('--stage', '-s', type=int, show_default='no stage')
@click.option('--data-tag', '-D', show_default='auto')
@click.option('--segment-tag', '-T', show_default='auto')
@click.option('--segment-stage', '-S', type=int, show_default='auto')
@model_options
@run_base
def temporal(obj):
    '''Update spike'''

    if obj.segment_tag is None:
        obj['segment_tag'] = obj.tag

    if obj.segment_stage is None:
        obj['segment_stage'] = obj.stage

    data = obj.data
    nt = obj.nt

    segment = obj.segment
    nk, nx = segment.shape

    variance = Variance(data, nk, nx, nt)
    variance.set_double_exp(**obj.tau)
    nu = variance.nu

    fp_input = Input(nk, nx, name='footprint')
    sp_input = Input(nk, nu, name='spike')

    model = SpikeModel(fp_input, sp_input, variance)
    model.set_penalty(**obj.reg)
    model.compile()
    log = model.fit(
        segment, **obj.opt,
        #log_dir=logs, stage=base,
    )

    obj.save_numpy(model.spike.val, 'spike')
    return dict(log=log.history)
