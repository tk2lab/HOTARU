import click
import numpy as np

from hotaru.optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from hotaru.train.variance import Variance

from hotaru.train.footprint import FootprintModel

from .base import run_base


@click.command()
@click.option('--batch', type=int, default=100, show_default=True)
@run_base
def spatial(obj, batch):
    '''Update footprint'''

    if (obj.prev_stage is None) and isinstance(obj.stage, int):
        obj.prev_stage = obj.stage - 1

    data = obj.data()
    spike = obj.spike()
    nk = spike.shape[0]
    nx = obj.nx()
    nt = obj.nt()

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
        batch=batch, verbose=obj.verbose,
        #log_dir=logs, stage=base,
    )

    obj.save_numpy(model.footprint.val, 'footprint')
    return dict(log=log.history)
