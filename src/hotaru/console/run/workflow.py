import click

from ..base import configure
from .clean import clean
from .data import data
from .find import find
from .make import make
from .spatial import spatial
from .temporal import temporal


@click.command(context_settings=dict(show_default=True))
@click.option("--tag", type=str, callback=configure, is_eager=True)
@click.option("--end-stage", type=int)
@click.option(
    "--end-kind", type=click.Choice(["spike", "footprint", "segment"])
)
@click.option("--non-stop", is_flag=True)
@click.option("--storage-saving", is_flag=True)
@click.pass_context
def workflow(ctx, tag, end_stage, end_kind, non_stop, storage_saving):
    """Workflow"""

    workflow_local(
        ctx, ctx.obj, tag, end_stage, end_kind, non_stop, storage_saving
    )


def workflow_local(
    ctx, obj, tag, end_stage, end_kind, non_stop, storage_saving
):
    prev_tag = obj.get_config("workflow", tag, "prev_tag")
    if prev_tag and (prev_tag != tag):
        prev_stage = int(obj.get_config("workflow", tag, "prev_stage"))
        prev_kind = obj.get_config("workflow", tag, "prev_kind")
        workflow_local(ctx, obj, prev_tag, prev_stage, prev_kind, True, False)
        click.echo(f"workflow {tag} {end_stage} {end_kind}")
    else:
        click.echo(f"workflow {tag} {end_stage} {end_kind}")
        prev_tag = tag
        find_tag = obj.get_config("make", prev_tag, "find_tag")
        data_tag = obj.get_config("find", find_tag, "data_tag")
        obj.invoke(ctx, data, f"--tag={data_tag}")
        obj.invoke(ctx, find, f"--tag={find_tag}")
        obj.invoke(ctx, make, f"--tag={prev_tag}")
        prev_stage = 0
        prev_kind = "segment"

    args = [f"--tag={tag}"]
    if storage_saving:
        args.append(f"--storage-saving")

    stage = 1
    if prev_kind == "segment":
        obj.invoke(
            ctx,
            temporal,
            f"--segment-tag={prev_tag}",
            f"--segment-stage={prev_stage}",
            *args,
        )
        kind = "spike"
    elif prev_kind == "spike":
        obj.invoke(
            ctx,
            spatial,
            f"--spike-tag={prev_tag}",
            f"--spike-stage={prev_stage}",
            *args,
        )
        kind = "footprint"
    elif prev_kind == "footprint":
        obj.invoke(
            ctx,
            clean,
            f"--footprint-tag={prev_tag}",
            f"--footprint-stage={prev_stage}",
            *args,
        )
        stage += 1
        kind = "segment"

    while (stage < end_stage) or (kind != end_kind):
        _s = 999 if storage_saving else stage
        if kind == "segment":
            obj.invoke(
                ctx,
                temporal,
                f"--segment-tag={tag}",
                f"--segment-stage={stage-1}",
                *args,
            )
            kind = "spike"
        elif kind == "spike":
            obj.invoke(
                ctx,
                spatial,
                f"--spike-tag={tag}",
                f"--spike-stage={stage}",
                *args,
            )
            kind = "footprint"
        elif kind == "footprint":
            obj.invoke(
                ctx,
                clean,
                f"--footprint-tag={tag}",
                f"--footprint-stage={stage}",
                *args,
            )
            stage += 1
            kind = "segment"
