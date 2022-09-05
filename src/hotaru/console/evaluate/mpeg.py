import click

from ...evaluate.make_mpeg import make_mpeg
from ..base import configure


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--stage", "-s", type=int)
@click.option("--has-truth", is_flag=True)
@click.pass_obj
def mpeg(obj, tag, stage, has_truth):
    """Make Mp4"""

    data_tag = obj.log("3segment", tag, stage)["data_tag"]
    if stage is None:
        stage = "curr"
    else:
        stage = f"{stage:03}"
    make_mpeg(data_tag, tag, stage, has_truth)
