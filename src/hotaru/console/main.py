from configparser import ConfigParser

import click

from .obj import Obj
from .run import run
#from .fig import fig

commands = dict(run=[
    'data', 'find', 'init', 'spatial', 'temporal', 'clean', 'auto',
])

def configure(ctx, param, filename):
    cfg = ConfigParser()
    cfg.read(filename)
    ctx.default_map = {}
    if 'DEFAULT' in cfg:
        ctx.default_map.update(cfg['DEFAULT'])
    if 'hotaru' in cfg:
        ctx.default_map.update(cfg['hotaru'])
    for cmdname in ['run']:
        defaults1 = ctx.default_map.setdefault(cmdname, {})
        if 'DEFAULT' in cfg:
            defaults1.update(cfg['DEFAULT'])
        for cmd in commands[cmdname]:
            defaults2 = defaults1.setdefault(cmd, {})
            if 'DEFAULT' in cfg:
                defaults2.update(cfg['DEFAULT'])


@click.group()
@click.option(
    '--config', '-c', type=click.Path(dir_okay=False), default='hotaru.ini',
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
#main.add_command(fig)
