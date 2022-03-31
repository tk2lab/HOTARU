import click

from .tau import tau_options
from .tau import tau_wrap
from .reg import reg_options
from .reg import reg_wrap
from .opt import opt_options
from .opt import opt_wrap

from ..obj import Obj
from .data import data
from .find import find
from .init import init
from .temporal import temporal
from .spatial import spatial
#from .clean import clean


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
    obj.data_tag = data_tag or tag
    obj.prev_tag = prev_tag or tag
    obj.stage = stage
    obj.prev_stage = prev_stage
    obj.tau = tau
    obj.reg = reg
    obj.opt = opt
    obj.force = force


run.add_command(data)
run.add_command(find)
run.add_command(init)
run.add_command(temporal)
run.add_command(spatial)
