import pkgutil
from configparser import ConfigParser

import click

from .obj import Obj
from .config import config
from .run.data import data
from .run.find import find
from .run.init import init
from .run.temporal import temporal
from .run.spatial import spatial
from .run.clean import clean
from .run.output import output
from .trial import trial
from .auto import auto
from .figure import figure
from .mpeg import mpeg
from .sample import sample


def configure(ctx, param, configfile):
    cfg = ConfigParser()
    default = pkgutil.get_data('hotaru.console', 'hotaru.ini').decode('utf-8')
    cfg.read_string(default)
    cfg.read(configfile)
    return cfg


@click.group()
@click.option(
    '--config', '-c', type=click.Path(dir_okay=False),
    default='hotaru.ini', callback=configure, show_default=True,
)
@click.option('--tag', '-t')
@click.option('--workdir', '-w', type=str)
@click.option('--data-tag', '-D')
@click.option('--find-tag', '-F')
@click.option('--init-tag', '-I')
@click.option('--force', '-f', is_flag=True)
@click.option('--quit', '-q', is_flag=True)
@click.option('--verbose', '-v', count=True)
@click.pass_context
def main(ctx, config, tag, **args):
    '''Main'''

    if tag is None:
        tag = config.get('main', 'tag')

    if tag in config:
        for opt, val in args.items():
            if val is None:
                args[opt] = config.get(tag, opt)

    ctx.ensure_object(Obj)
    obj = ctx.obj
    obj.config = config
    obj['tag'] = tag
    obj.update(args)
    if obj.data_tag == '':
        obj['data_tag'] = obj.tag
    if obj.find_tag == '':
        obj['find_tag'] = obj.tag
    if obj.init_tag == '':
        obj['init_tag'] = obj.tag
    obj['verbose'] = 0 if obj.quit else (obj.verbose or 1)
    del obj['quit']


main.add_command(config)
main.add_command(data)
main.add_command(find)
main.add_command(init)
main.add_command(temporal)
main.add_command(spatial)
main.add_command(clean)
main.add_command(output)
main.add_command(trial)
main.add_command(auto)
main.add_command(figure)
main.add_command(mpeg)
main.add_command(sample)
