import click
import numpy as np

from hotaru.optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from hotaru.train.variance import Variance

from hotaru.train.spike import SpikeModel

from .base import run_base


@click.command()
@click.option('--initial', is_flag=True)
@click.option('--batch', type=int, default=100, show_default=True)
@run_base
def temporal(obj, initial, batch):
    '''Update spike'''

    if obj.prev_stage is None:
        obj.prev_stage = obj.stage

    if obj.prev_stage == 0:
        initial = True

    data = obj.data()
    nt = obj.nt()

    segment = obj.segment(initial)
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
        segment, **obj.opt, batch=batch,
        #log_dir=logs, stage=base,
    )

    obj.save_numpy(model.spike.val, 'spike')
    return dict(log=log.history)
