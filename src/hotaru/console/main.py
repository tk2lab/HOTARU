from configparser import ConfigParser

import click

from .obj import Obj
from .run.data import data
from .run.find import find
from .run.init import init
from .run.temporal import temporal
from .run.spatial import spatial
from .run.clean import clean
from .run.output import output
from .auto import auto
#from .fig import fig


def configure(ctx, param, filename):
    cfg = ConfigParser()
    cfg.read(filename)
    return cfg


@click.group()
@click.option(
    '--config', '-c', type=click.Path(dir_okay=False),
    default='hotaru.ini', callback=configure, show_default=True,
)
@click.option('--workdir', '-w', type=str, default='hotaru', show_default=True)
@click.option('--tag', '-t', default='default', show_default=True)
@click.option('--force', '-f', is_flag=True)
@click.option('--quit', '-q', is_flag=True)
@click.option('--verbose', '-v', count=True)
@click.pass_context
def main(ctx, config, **args):
    '''Main'''

    ctx.ensure_object(Obj)
    obj = ctx.obj
    obj.config = config

    args['verbose'] = 0 if quit else (args.verbose or 1)
    del args['quit']

    sec = 'main'
    if obj.config.has_section(sec):
        for key in args.keys():
            if obj.config.has_option(sec, key):
                args[key] = obj.config.get(sec, key)

    obj.update(args)


main.add_command(data)
main.add_command(find)
main.add_command(init)
main.add_command(temporal)
main.add_command(spatial)
main.add_command(clean)
main.add_command(output)
main.add_command(auto)
'''
main.add_command(figure)
main.add_command(trial)
'''
