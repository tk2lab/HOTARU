from functools import wraps

import click

from hotaru.util.timer import Timer


def run_command(*options, pass_context=False):
    def deco(command):
        @click.command()
        @wraps(command)
        @click.pass_context
        def new_command(ctx, **args):
            ctx.obj["kind"] = command.__name__

            for opt, val in args.items():
                if val is None:
                    args[opt] = types[opt](
                        ctx.obj.config.get(ctx.obj.tag, opt)
                    )
            ctx.obj.update(args)

            if not ctx.obj.need_exec():
                return

            click.echo(f"{ctx.obj.kind}, {ctx.obj.tag}: {args}")
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
