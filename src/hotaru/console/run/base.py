from functools import wraps

import click

from hotaru.util.timer import Timer

from ..obj import Obj


def run_command(*options):

    def deco(command):

        @click.command()
        @wraps(command)
        @click.pass_obj
        def new_command(obj, **args):
            obj['kind'] = command.__name__

            if obj.tag in obj.config:
                for opt, val in args.items():
                    if val is None:
                        args[opt] = types[opt](obj.config.get(obj.tag, opt))
            obj.update(args)

            if not obj.need_exec():
                return

            click.echo(f'{obj.kind}, {obj.tag}: {args}')
            with Timer() as timer:
                log = command(obj)

            if log is not None:
                log['time'] = timer.get()
                log.update(args)
                obj.save_log(log)
                click.echo(f'time: {timer.get()[0]}')

        types = {o.name: o.type for o in options}

        for opt in options:
            new_command.params.append(opt)

        return new_command

    return deco
