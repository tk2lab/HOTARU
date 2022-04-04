import click

from hotaru.optimizer.input import MaxNormNonNegativeL1InputLayer as Input
from hotaru.train.variance import Variance
from hotaru.train.spike import SpikeModel

from .base import run_command
from .options import model_options


@run_command(
    click.Option(['--stage', '-s'], type=int),
    click.Option(['--segment-tag', '-T']),
    click.Option(['--segment-stage', '-S'], type=int),
    *model_options(),
)
def temporal(obj):
    '''Update spike'''

    if obj.stage == -1:
        obj['stage'] = '_curr'

    if obj.segment_tag == '':
        if obj.stage == 0:
            obj['segment_tag'] = obj.init_tag
        else:
            obj['segment_tag'] = obj.tag

    if obj.segment_stage == -1:
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
    return dict(history=log.history)
