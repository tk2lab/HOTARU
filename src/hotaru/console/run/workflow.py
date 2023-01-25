import click

from ..base import configure
from .data import data
from .find import find
from .make import make
from .spatial import spatial
from .temporal import temporal

_kind_choice = click.Choice(["temporal", "spatial"])
_next_kind = dict(temporal="spatial", spatial="temporal")


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--end-stage", type=int)
@click.option("--end-kind", type=_kind_choice)
@click.option("--non-stop", is_flag=True)
@click.pass_context
def workflow(ctx, tag, end_stage, end_kind, non_stop):
    """Workflow"""

    get_config = lambda *x: ctx.obj.get_config(*x)
    invoke = lambda *x: ctx.obj.invoke(ctx, *x)
    workflow_local(get_config, invoke, tag, end_stage, end_kind, non_stop)


def workflow_local(get_config, invoke, tag, end_stage, end_kind, non_stop):
    prev_tag = get_config("workflow", tag, "prev_tag")
    if prev_tag and (prev_tag != tag):
        prev_kind = get_config("workflow", tag, "prev_kind")
        prev_stage = int(get_config("workflow", tag, "prev_stage"))
        workflow_local(
            get_config, invoke, prev_tag, prev_stage, prev_kind, True
        )
        click.echo(f"workflow {tag} {end_stage} {end_kind}")
        prev_tag = tag
        if prev_kind == "data":
            find_tag = get_config("make", prev_tag, "find_tag") or prev_tag
            invoke(find, f"--tag={find_tag}")
            prev_kind = "find"
        if prev_kind == "find":
            invoke(make, f"--tag={prev_tag}")
            prev_kind = "spatial"
            prev_stage = 1
        kind, stage = workflow_step(
            invoke, tag, prev_tag, prev_kind, prev_stage
        )
    else:
        click.echo(f"workflow {tag} {end_stage} {end_kind}")
        prev_tag = tag
        find_tag = get_config("make", prev_tag, "find_tag") or prev_tag
        data_tag = get_config("find", find_tag, "data_tag") or find_tag
        invoke(data, f"--tag={data_tag}")
        if end_kind == "data":
            return
        invoke(find, f"--tag={find_tag}")
        if end_kind == "find":
            return
        invoke(make, f"--tag={prev_tag}")
        if end_kind == "make":
            return
        kind, stage = "spatial", 1

    while (stage < end_stage) or (kind != end_kind):
        kind, stage = workflow_step(invoke, tag, tag, kind, stage)


def workflow_step(invoke, tag, prev_tag, prev_kind, prev_stage):
    kind = _next_kind[prev_kind]
    if tag != prev_tag:
        stage = 1
    else:
        stage = prev_stage
        if kind == "spatial":
            stage += 1
    if kind == "spatial":
        invoke(
            spatial,
            f"--tag={tag}",
            f"--temporal-tag={prev_tag}",
            f"--temporal-stage={prev_stage}",
        )
    elif kind == "temporal":
        invoke(
            temporal,
            f"--tag={tag}",
            f"--spatial-tag={prev_tag}",
            f"--spatial-stage={prev_stage}",
        )
    return kind, stage
