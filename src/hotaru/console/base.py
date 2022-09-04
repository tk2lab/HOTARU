from functools import wraps

import click

from ..util.timer import Timer


def run_command(*options, pass_context=False):
    def deco(command):
        @click.command()
        @wraps(command)
        @click.pass_context
        def new_command(ctx, **args):
            kind = command.__name__

            exec_tag = ctx.obj.tag
            if kind == "data":
                exec_tag = ctx.obj.data_tag
            elif kind == "find":
                exec_tag = ctx.obj.find_tag
            elif kind == "init":
                exec_tag = ctx.obj.init_tag

            ctx.obj["kind"] = kind
            ctx.obj["exec_tag"] = exec_tag
            for opt, val in args.items():
                if val is None:
                    args[opt] = types[opt](
                        ctx.obj.config.get(exec_tag, opt)
                    )
            ctx.obj.update(args)

            if not ctx.obj.need_exec():
                click.echo(f"skip: {kind}, {exec_tag}")
                return

            click.echo("-----------------------------------")
            click.echo(f"{kind}, {exec_tag}:")
            for k, v in args.items():
                click.echo(f"   {k}: {v}")

            with Timer() as timer:
                log = command(ctx if pass_context else ctx.obj)

            if log is not None:
                log["time"] = timer.get()
                log.update(args)
                ctx.obj.save_log(log)
                click.echo(f"time: {timer.get()[0]}")

        types = {o.name: o.type for o in options}

        for opt in options:
            new_command.params.append(opt)

        return new_command

    return deco
