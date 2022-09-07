import click

from ..base import configure
from .clean import clean
from .data import data
from .find import find
from .make import make
from .spatial import spatial
from .temporal import temporal


def workflow_local(ctx, obj, tag, stage, save, non_stop):
    click.echo(f"workflow {tag} {stage}")
    prev_tag = obj.get_config("workflow", tag, "prev_tag")
    if prev_tag:
        prev_stage = obj.get_config("workflow", tag, "prev_stage")
        workflow_local(ctx, obj, prev_tag, prev_stage, True, True)
    else:
        prev_tag = tag
        find_tag = obj.get_config("make", prev_tag, "find_tag")
        data_tag = obj.get_config("find", find_tag, "data_tag")
        obj.invoke(ctx, data, f"--tag={data_tag}")
        obj.invoke(ctx, find, f"--tag={find_tag}")
        obj.invoke(ctx, make, f"--tag={prev_tag}")

    args = [f"--tag={tag}"]
    if save:
        args.append(f"--storing-intermidiate-results")
    obj.invoke(
        ctx, temporal, f"--segment-tag={prev_tag}", "--segment-stage=0", *args
    )
    for s in range(1, stage + 1):
        _s = s if save else 999
        obj.invoke(ctx, spatial, f"--tag={tag}", f"--stage={_s}")
        obj.invoke(ctx, clean, f"--tag={tag}", f"--stage={_s}")
        obj.invoke(
            ctx,
            temporal,
            f"--segment-tag={tag}",
            f"--segment-stage={_s}",
            *args,
        )


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--storing-intermidiate-results", is_flag=True, default=None)
@click.option("--max-stage", type=int)
@click.option("--non-stop", is_flag=True, default=None)
@click.pass_context
def workflow(ctx, tag, storing_intermidiate_results, max_stage, non_stop):
    """Workflow"""

    save = storing_intermidiate_results
    workflow_local(ctx, ctx.obj, tag, max_stage, save, non_stop)
