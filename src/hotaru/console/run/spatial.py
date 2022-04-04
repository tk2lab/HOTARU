import click

from hotaru.optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from hotaru.train.variance import Variance
from hotaru.train.footprint import FootprintModel

from .base import run_base
from .options import model_options


@click.command()
@click.option('--stage', '-s', type=int, show_default='no stage')
@click.option('--data-tag', '-D', show_default='auto')
@click.option('--spike-tag', '-T', show_default='auto')
@click.option('--spike-stage', '-S', type=int, show_default='auto')
@model_options
@click.option('--batch', type=int, default=100, show_default=True)
@run_base
def spatial(obj):
    '''Update footprint'''

    if obj.spike_tag is None:
        obj['spike_tag'] = obj.tag
    if obj.spike_stage is None:
        obj['spike_stage'] = obj.stage

    data = obj.data
    spike = obj.spike
    nk = spike.shape[0]
    nx = obj.nx
    nt = obj.nt

    variance = Variance(data, nk, nx, nt)
    variance.set_double_exp(**obj.tau)
    nu = variance.nu

    fp_input = Input(nk, nx, name='footprint')
    sp_input = Input(nk, nu, name='spike')

    model = FootprintModel(fp_input, sp_input, variance)
    model.set_penalty(**obj.reg)
    model.compile()
    log = model.fit(
        spike, **obj.opt,
        #log_dir=logs, stage=base,
    )

    obj.save_numpy(model.footprint.val, 'footprint')
    return dict(log=log.history)
