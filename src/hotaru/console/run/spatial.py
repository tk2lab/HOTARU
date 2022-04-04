import click

from hotaru.optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from hotaru.train.variance import Variance
from hotaru.train.footprint import FootprintModel

from .base import run_command
from .options import model_options


@run_command(
    click.Option(['--stage', '-s'], type=int),
    click.Option(['--spike-tag', '-T']),
    click.Option(['--spike-stage', '-S'], type=int),
    *model_options(),
    click.Option(['--batch'], type=int, default=100, show_default=True),
)
def spatial(obj):
    '''Update footprint'''

    if obj.stage == -1:
        obj['stage'] = '_curr'

    if obj.spike_tag == '':
        obj['spike_tag'] = obj.tag

    if obj.spike_stage == -1:
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
    return dict(history=log.history)
