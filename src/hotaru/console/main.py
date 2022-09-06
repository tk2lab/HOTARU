import configparser
import pkgutil

import click

from .config.config import config
from .evaluate.figure import figure
from .evaluate.mpeg import mpeg
from .evaluate.output import output
from .evaluate.sample import sample
from .evaluate.stats import stats
from .evaluate.tune import tune
from .obj import Obj
from .run.clean import clean
from .run.data import data
from .run.find import find
from .run.make import make
from .run.spatial import spatial
from .run.temporal import temporal
from .run.workflow import workflow


class NatureOrderGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()


def configure(ctx, param, configfile):
    config = configparser.ConfigParser(
        allow_no_value=True,
        interpolation=configparser.ExtendedInterpolation(),
    )
    default = pkgutil.get_data("hotaru.console.config", "default.ini")
    config.read_string(default.decode("utf-8"))
    config.read(configfile)
    ctx.default_map = config["main"]
    ctx.obj = ctx.with_resource(Obj())
    ctx.obj.config = config


@click.group(cls=NatureOrderGroup)
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
@click.option("--force", "-f", is_flag=True)
@click.pass_context
def main(ctx, workdir, force):
    """Main"""

    ctx.obj.workdir = workdir
    ctx.obj.force = force


@click.group(cls=NatureOrderGroup)
def run():
    """Step Run"""


@click.group()
def output():
    """Output Results"""


main.add_command(config)
main.add_command(stats)
main.add_command(tune)
main.add_command(workflow)
main.add_command(output)
main.add_command(run)
main.add_command(sample)

run.add_command(data)
run.add_command(find)
run.add_command(make)
run.add_command(temporal)
run.add_command(spatial)
run.add_command(clean)

output.add_command(output)
output.add_command(figure)
output.add_command(mpeg)
