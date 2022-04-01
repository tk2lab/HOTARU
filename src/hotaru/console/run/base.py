from functools import wraps

import click

from hotaru.util.timer import Timer

from ..obj import Obj
from .radius import radius_options
from .radius import radius_wrap
from .tau import tau_options
from .tau import tau_wrap
from .reg import reg_options
from .reg import reg_wrap
from .opt import opt_options
from .opt import opt_wrap


@click.group()
@click.option('--tag', '-t', default='default', show_default=True)
@click.option('--stage', '-s', type=int, show_default='no stage')
@click.option('--prev-tag', '-T', show_default='auto')
@click.option('--prev-stage', '-S', type=int, show_default='auto')
@click.option('--data-tag', '-D', show_default='auto')
@click.option('--find-tag', '-F', show_default='auto')
@click.option('--init-tag', '-I', show_default='auto')
@click.option(
    '--imgs-path',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default='imgs.tif',
    show_default=True,
)
@click.option(
    '--mask-type',
    default='0.pad',
    show_default=True,
)
@radius_options
@click.option('--distance', type=float, default=1.6, show_default=True)
@tau_options
@reg_options
@opt_options
@click.option('--force', '-f', is_flag=True)
@radius_wrap
@tau_wrap
@reg_wrap
@opt_wrap
@click.pass_obj
def run(obj, tag, stage, prev_tag, prev_stage, data_tag, find_tag, init_tag, imgs_path, mask_type, radius, distance, tau, reg, opt, force):
    '''Run'''

    obj.tag = tag
    obj.stage = stage
    obj.force = force

    obj.prev_tag = prev_tag or tag
    obj.prev_stage = prev_stage

    obj.data_tag = data_tag or tag
    obj.find_tag = find_tag or tag
    obj.init_tag = init_tag or tag

    obj.imgs_path = imgs_path
    obj.mask_type = mask_type
    obj.radius = radius
    obj.distance = distance
    obj.tau = tau
    obj.reg = reg
    obj.opt = opt


def run_base(func):

    @wraps(func)
    @click.pass_obj
    def new_func(obj, **args):
        click.echo(func.__name__)
        if not obj.need_exec(func.__name__):
            return

        with Timer() as timer:
            save_prev_stage = obj.prev_stage
            log = func(obj, **args)
            obj.prev_stage = save_prev_stage

        log.update(args)
        log['time'] = timer.get()
        obj.save_log(func.__name__, log)

    return new_func
