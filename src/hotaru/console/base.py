import functools

import click

from ..util.timer import Timer

readable_file = click.Path(exists=True, dir_okay=False, readable=True)


def configure(ctx, param, tag):
    if tag is None:
        tag = ctx.obj.config.get("main", "tag")
    section = ctx.info_name
    ctx.default_map = ctx.obj.config.defaults()
    if section in ctx.obj.config:
        ctx.default_map.update(ctx.obj.config[section])
    section = f"{ctx.info_name}/{tag}"
    if section in ctx.obj.config:
        ctx.default_map.update(ctx.obj.config[section])
    return tag


def command_wrap(command):
    @functools.wraps(command)
    def wraped_command(obj, tag, **args):
        kind = command.__name__

        click.echo("-----------------------------------")
        click.echo(f"{kind}, {tag}:")

        if obj.can_skip(kind, tag, **args):
            click.echo("    skip")
            return

        for k, v in args.items():
            click.echo(f"   {k}: {v}")

        with Timer() as timer:
            log = command(obj, tag, **args)
        time = timer.get()
        click.echo(f"time: {time[0]}")

        if log:
            log, kind, tag, stage = log
            args.update(log)
            args["time"] = time
            obj.save_log(args, kind, tag, stage)

    return wraped_command
