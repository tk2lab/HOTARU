from configparser import ConfigParser

import click

from .obj import Obj
from .run import run


def configure(ctx, param, filename):
    cfg = ConfigParser()
    cfg.read(filename)
    ctx.default_map = {}
    if 'DEFAULT' in cfg:
        ctx.default_map.update(cfg['DEFAULT'])
    if 'hotaru' in cfg:
        ctx.default_map.update(cfg['hotaru'])
    for cmdname in ['data']:
        defaults = ctx.default_map.setdefault(cmdname, {})
        if 'DEFAULT' in cfg:
            defaults.update(cfg['DEFAULT'])
        if f'hotaru.{cmdname}' in cfg:
            defaults.update(cfg[f'hotaru.{cmdname}'])


@click.group()
@click.option(
    '--config', '-c', type=click.Path(dir_okay=False), default='hotaru/config.ini',
    callback=configure, is_eager=True, expose_value=False, show_default=True,
    help='',
)
@click.option('--workdir', '-w', type=str, default='hotaru', show_default=True)
@click.option('--quit', '-q', is_flag=True)
@click.option('--verbose', '-v', count=True)
@click.pass_context
def main(ctx, workdir, quit, verbose):
    '''Main'''

    ctx.ensure_object(Obj)
    ctx.obj.workdir = workdir
    ctx.obj.verbose = 0 if quit else (verbose or 1)


main.add_command(run)
