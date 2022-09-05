import functools

import click

from ..util.timer import Timer


def configure(ctx, param, tag):
    tag = tag or ctx.obj.tag
    section = f"{ctx.info_name}/{tag}"
    if section in ctx.obj.config:
        ctx.default_map = ctx.obj.config[section]
    else:
        ctx.default_map = ctx.obj.config.defaults()
    return tag


def command_wrap(command):
    @functools.wraps(command)
    def wraped_command(obj, tag, **args):
        kind = command.__name__

        if obj.can_skip("1data", tag, 0):
            click.echo(f"skip: {kind}, {tag}")
            return

        click.echo("-----------------------------------")
        click.echo(f"{kind}, {tag}:")
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
