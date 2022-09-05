import configparser
import pkgutil

import click

# from .driver.auto import auto
from .driver.config import config

# from .driver.trial import trial
# from .evaluate.figure import figure
from .evaluate.mpeg import mpeg
from .evaluate.output import output
from .evaluate.sample import sample
from .evaluate.stats import stats
from .obj import Obj
from .run.clean import clean
from .run.data import data
from .run.find import find
from .run.make import make
from .run.spatial import spatial
from .run.temporal import temporal


def configure(ctx, param, configfile):
    config = configparser.ConfigParser(
        allow_no_value=True,
        interpolation=configparser.ExtendedInterpolation(),
    )
    default = pkgutil.get_data("hotaru.console", "hotaru.ini")
    config.read_string(default.decode("utf-8"))
    config.read(configfile)
    ctx.default_map = config["main"]
    ctx.obj = ctx.with_resource(Obj())
    ctx.obj.config = config


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(dir_okay=False),
    default="hotaru.ini",
    callback=configure,
    is_eager=True,
    expose_value=False,
    show_default=True,
)
@click.option("--workdir", "-w", type=str)
@click.option("--tag", "-t", type=str)
@click.option("--force", "-f", is_flag=True)
@click.pass_context
def main(ctx, workdir, tag, force):
    """Main"""

    ctx.obj.workdir = workdir
    ctx.obj.tag = tag
    ctx.obj.force = force


main.add_command(sample)

main.add_command(config)
main.add_command(stats)
# main.add_command(trial)
# main.add_command(auto)

main.add_command(data)
main.add_command(find)
main.add_command(make)
main.add_command(temporal)
main.add_command(spatial)
main.add_command(clean)

main.add_command(output)
main.add_command(mpeg)

# main.add_command(figure)
