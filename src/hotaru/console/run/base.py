from functools import wraps

import click

from hotaru.util.timer import Timer

from ..obj import Obj
from .tau import tau_options
from .tau import tau_wrap
from .reg import reg_options
from .reg import reg_wrap
from .opt import opt_options
from .opt import opt_wrap


def run_base(func):

    @wraps(func)
    @click.pass_obj
    def new_func(obj, **args):
        click.echo(func.__name__)
        if not obj.need_exec(func.__name__):
            return

        with Timer() as timer:
            log = func(obj, **args)

        log.update(args)
        log['time'] = timer.get()
        obj.save_log(func.__name__, log)

    return new_func


@click.group()
@click.option('--tag', '-t', default='default', show_default=True)
@click.option('--stage', '-s', type=int, show_default='no stage')
@click.option('--data-tag', '-D', show_default='auto')
@click.option('--prev-tag', '-T', show_default='auto')
@click.option('--prev-stage', '-S', type=int, show_default='auto')
@tau_options
@reg_options
@opt_options
@click.option('--force', '-f', is_flag=True)
@tau_wrap
@reg_wrap
@opt_wrap
@click.pass_obj
def run(obj, tag, stage, data_tag, prev_tag, prev_stage, tau, reg, opt, force):
    '''Run'''

    obj.tag = tag
    obj.stage = stage
    obj.force = force

    obj.data_tag = data_tag or tag
    obj.prev_tag = prev_tag or tag
    obj.prev_stage = prev_stage

    obj.tau = tau
    obj.reg = reg
    obj.opt = opt
